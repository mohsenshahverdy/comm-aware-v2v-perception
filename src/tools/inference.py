# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import json
import csv
import hashlib
from itertools import islice
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

import src.hypes_yaml.yaml_utils as yaml_utils
from src.tools import train_utils, inference_utils
from src.data_utils.datasets import build_dataset
from src.utils import eval_utils
from src.utils.logging import get_logger
from src.utils.runtime_config import apply_runtime_overrides, log_and_validate_communication_approach, set_global_seed
import matplotlib.pyplot as plt


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(x, default=0.0):
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _append_jsonl(path, payload):
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _config_fingerprint(hypes):
    return hashlib.sha256(str(hypes).encode("utf-8")).hexdigest()



def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str, choices=['intermediate'],
                        default='intermediate',
                        help='fusion mode (current repo dataset registry supports only intermediate)')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Optional sample cap for quick smoke tests (0 means full split).')
    parser.add_argument('--skip_ap', action='store_true',
                        help='Skip AP accumulation/final eval and only test forward + communication metrics.')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Override root_dir from YAML (train split path)')
    parser.add_argument('--validate_dir', type=str, default=None,
                        help='Override validate_dir from YAML (test/validation split path)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed override for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic CUDA algorithms where possible')
    parser.add_argument('--benchmark', dest='benchmark', action='store_true', default=None,
                        help='Enable cuDNN benchmark (defaults from reproducibility config)')
    parser.add_argument('--no-benchmark', dest='benchmark', action='store_false',
                        help='Disable cuDNN benchmark')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    logger = get_logger("Inference")
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    if opt.show_sequence:
        try:
            import open3d as o3d
            from src.visualization import vis_utils
        except ImportError as e:
            raise ImportError("`--show_sequence` requires `open3d`. Install it or run without sequence visualization.") from e

    hypes = yaml_utils.load_yaml(None, opt)
    log_and_validate_communication_approach(hypes, logger=logger)
    seed, deterministic, benchmark = apply_runtime_overrides(hypes, opt, logger=logger)
    set_global_seed(seed, deterministic=deterministic, benchmark=benchmark, logger=logger)
    comm_logging_cfg = hypes.get("communication", {}).get("logging", {})
    save_csv = bool(comm_logging_cfg.get("save_csv", True))
    save_per_frame_json = bool(comm_logging_cfg.get("save_per_frame_json", True))

    logger.step("Building dataset")
    src_dataset = build_dataset(hypes, visualize=True, train=False)
    logger.info("Dataset ready", samples=len(src_dataset))
    num_workers = 4
    data_loader = DataLoader(src_dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=src_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    logger.step("Creating model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        logger.run("Using GPU for inference")
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.step("Loading checkpoint")
    saved_path = opt.model_dir
    rr_cfg = hypes.get("communication", {}).get("receiver_request", {})
    if bool(rr_cfg.get("save_request_maps", False)) and hasattr(model, "comm_policy"):
        model.comm_policy.update_debug_dir(os.path.join(saved_path, "receiver_request_debug"))
        logger.config(
            "Receiver-request debug export enabled",
            debug_dir=os.path.join(saved_path, "receiver_request_debug"),
            debug_num_frames=rr_cfg.get("debug_num_frames", 5),
        )
    loaded_epoch, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())
    total_samples = len(data_loader)
    if opt.max_samples and opt.max_samples > 0:
        total_samples = min(total_samples, int(opt.max_samples))
        logger.info("Smoke sample limit enabled", max_samples=total_samples)
    logger.run("Starting inference", total_samples=total_samples)
    frame_jsonl_path = os.path.join(opt.model_dir, "comm_metrics_frame.jsonl")
    trace_jsonl_path = os.path.join(opt.model_dir, "inference_trace.jsonl")
    epoch_csv_path = os.path.join(opt.model_dir, "comm_metrics_epoch.csv")
    run_info_path = os.path.join(opt.model_dir, "inference_run_info.json")
    event_log_path = os.path.join(opt.model_dir, "inference_events.log")
    comm_stats_list = []

    file_logger = get_logger("Inference", log_to_file=True, file_path=event_log_path)
    def _event(level, msg, **fields):
        if level == "config":
            file_logger.config(msg, **fields)
        elif level == "metric":
            file_logger.metric(msg, **fields)
        elif level == "warn":
            file_logger.warn(msg, **fields)
        elif level == "save":
            file_logger.save(msg, **fields)
        elif level == "success":
            file_logger.success(msg, **fields)
        else:
            file_logger.run(msg, **fields)

    run_info = {
        "mode": "inference",
        "created_at_utc": _utc_now(),
        "model_dir": saved_path,
        "fusion_method": opt.fusion_method,
        "checkpoint_epoch_loaded": int(loaded_epoch) if loaded_epoch is not None else None,
        "global_sort_detections": bool(opt.global_sort_detections),
        "dataset_size": len(src_dataset),
        "device": str(device),
        "config_fingerprint_sha256": _config_fingerprint(hypes),
    }
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    _event("run", "Inference initialized", model_dir=saved_path)
    _event("run", "Checkpoint loaded", epoch=loaded_epoch)

    if save_csv and not os.path.exists(epoch_csv_path):
        with open(epoch_csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch", "train_loss", "val_loss", "ap_50", "ap_70",
                "feature_bytes_per_frame", "context_bytes_per_frame", "metadata_bytes_per_frame", "total_bytes_per_frame",
                "normalized_ratio", "feature_normalized_ratio", "context_normalized_ratio", "metadata_normalized_ratio", "total_normalized_ratio",
                "active_ratio", "active_neighbors_ratio",
                "packet_loss_rate", "receiver_request_keep_ratio",
                "receiver_request_context_ratio", "receiver_request_mask_metadata_ratio",
                "fps"
            ])
    inf_start = time.time()
    processed_frames = 0
    for i, batch_data in tqdm(enumerate(islice(data_loader, total_samples)), total=total_samples):
        frame_st = time.time()
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          src_dataset,
                                                          return_output_dict=True)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           src_dataset,
                                                           return_output_dict=True)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  src_dataset,
                                                                  return_output_dict=True)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            # collect communication stats from model output if available
            ego_out = {}
            if isinstance(output_dict, dict):
                if 'ego' in output_dict:
                    ego_out = output_dict['ego']
                elif len(output_dict) > 0:
                    first_key = next(iter(output_dict.keys()))
                    ego_out = output_dict[first_key]
            comm_stats = ego_out.get('comm_stats', {})
            if isinstance(comm_stats, dict) and len(comm_stats) > 0:
                comm_stats_list.append(comm_stats)
                if save_per_frame_json:
                    _append_jsonl(frame_jsonl_path, {"frame_idx": int(i), **{k: _safe_float(v) for k, v in comm_stats.items()}})

            if not opt.skip_ap:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat,
                                           0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat,
                                           0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat,
                                           0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                src_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=src_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

            frame_sec = max(time.time() - frame_st, 1e-6)
            frame_payload = {
                "ts_utc": _utc_now(),
                "frame_idx": int(i),
                "frame_seconds": _safe_float(frame_sec),
                "frame_fps": _safe_float(1.0 / frame_sec),
            }
            if isinstance(comm_stats, dict) and len(comm_stats) > 0:
                frame_payload["comm_stats"] = {k: _safe_float(v) for k, v in comm_stats.items()}
            if save_per_frame_json:
                _append_jsonl(trace_jsonl_path, frame_payload)
            processed_frames += 1

    if not opt.skip_ap:
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir,
                                      opt.global_sort_detections)
    else:
        logger.warn("AP evaluation skipped", reason="--skip_ap")

    # Merge AP + communication summary
    eval_path = os.path.join(opt.model_dir, 'eval_global_sort.yaml' if opt.global_sort_detections else 'eval.yaml')
    summary = yaml_utils.load_yaml(eval_path) if os.path.exists(eval_path) else {}
    if len(comm_stats_list) > 0:
        def _avg(key, default=0.0):
            vals = [float(s.get(key, default)) for s in comm_stats_list]
            return float(sum(vals) / max(len(vals), 1))
        summary.update({
            "comm_feature_bytes_per_frame": _avg("feature_bytes_per_frame", _avg("bytes_per_frame", 0.0)),
            "comm_context_bytes_per_frame": _avg("context_bytes_per_frame", 0.0),
            "comm_metadata_bytes_per_frame": _avg("metadata_bytes_per_frame", 0.0),
            "comm_total_bytes_per_frame": _avg("total_bytes_per_frame", _avg("bytes_per_frame", 0.0)),
            "comm_normalized_ratio": _avg("normalized_ratio", 1.0),
            "comm_feature_normalized_ratio": _avg("feature_normalized_ratio", _avg("normalized_ratio", 1.0)),
            "comm_context_normalized_ratio": _avg("context_normalized_ratio", 0.0),
            "comm_metadata_normalized_ratio": _avg("metadata_normalized_ratio", 0.0),
            "comm_total_normalized_ratio": _avg("total_normalized_ratio", _avg("normalized_ratio", 1.0)),
            "comm_active_ratio": _avg("active_ratio", 1.0),
            "comm_active_neighbors_ratio": _avg("active_neighbors_ratio", 1.0),
            "comm_packet_loss_rate": _avg("packet_loss_rate", 0.0),
            "receiver_request_keep_ratio": _avg("receiver_request_keep_ratio", 1.0),
            "receiver_request_context_ratio": _avg("receiver_request_context_ratio", 0.0),
            "receiver_request_mask_metadata_ratio": _avg("receiver_request_mask_metadata_ratio", 0.0),
        })
        # Backward compatibility
        summary["comm_bytes_per_frame"] = summary["comm_total_bytes_per_frame"]
        summary["comm_normalized_ratio"] = summary["comm_total_normalized_ratio"]
        if save_csv:
            with open(epoch_csv_path, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    "inference", None, None,
                    summary.get('ap_50', None), summary.get('ap_70', None),
                    summary["comm_feature_bytes_per_frame"],
                    summary["comm_context_bytes_per_frame"],
                    summary["comm_metadata_bytes_per_frame"],
                    summary["comm_total_bytes_per_frame"],
                    summary["comm_normalized_ratio"],
                    summary["comm_feature_normalized_ratio"],
                    summary["comm_context_normalized_ratio"],
                    summary["comm_metadata_normalized_ratio"],
                    summary["comm_total_normalized_ratio"],
                    summary["comm_active_ratio"],
                    summary["comm_active_neighbors_ratio"],
                    summary["comm_packet_loss_rate"],
                    summary["receiver_request_keep_ratio"],
                    summary["receiver_request_context_ratio"],
                    summary["receiver_request_mask_metadata_ratio"],
                    None
                ])
    yaml_utils.save_yaml(summary, os.path.join(opt.model_dir, "summary_eval.yaml"))

    total_sec = max(time.time() - inf_start, 1e-6)
    infer_fps = float(max(processed_frames, 1) / total_sec)
    compact = {
        "completed_at_utc": _utc_now(),
        "inference_seconds": _safe_float(total_sec),
        "inference_fps": _safe_float(infer_fps),
        "processed_frames": int(processed_frames),
        "max_samples": int(total_samples),
        "ap_30": summary.get("ap30", summary.get("ap_30", None)),
        "ap_50": summary.get("ap_50", None),
        "ap_70": summary.get("ap_70", None),
        "comm_total_bytes_per_frame": summary.get("comm_total_bytes_per_frame", None),
        "comm_feature_bytes_per_frame": summary.get("comm_feature_bytes_per_frame", None),
        "comm_context_bytes_per_frame": summary.get("comm_context_bytes_per_frame", None),
        "comm_metadata_bytes_per_frame": summary.get("comm_metadata_bytes_per_frame", None),
        "comm_normalized_ratio": summary.get("comm_normalized_ratio", None),
        "comm_feature_normalized_ratio": summary.get("comm_feature_normalized_ratio", None),
        "comm_context_normalized_ratio": summary.get("comm_context_normalized_ratio", None),
        "comm_metadata_normalized_ratio": summary.get("comm_metadata_normalized_ratio", None),
        "comm_total_normalized_ratio": summary.get("comm_total_normalized_ratio", None),
    }
    with open(os.path.join(opt.model_dir, "inference_summary.json"), "w") as f:
        json.dump(compact, f, indent=2)
    run_info.update(compact)
    run_info["completed"] = True
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    _event(
        "metric",
        "Inference finished",
        ap30=summary.get("ap30", summary.get("ap_30", "NA")),
        ap50=summary.get("ap_50", "NA"),
        ap70=summary.get("ap_70", "NA"),
        comm_normalized_ratio=summary.get("comm_normalized_ratio", "NA"),
        fps=f"{infer_fps:.3f}",
    )
    _event("save", "Summary saved", summary_path=os.path.join(opt.model_dir, "summary_eval.yaml"))

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
