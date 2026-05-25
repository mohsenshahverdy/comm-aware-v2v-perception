# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import json
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import src.hypes_yaml.yaml_utils as yaml_utils
from src.tools import train_utils, inference_utils
from src.data_utils.datasets import build_dataset
from src.utils import eval_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
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
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
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

    print('Dataset Building')
    src_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(src_dataset)} samples found.")
    num_workers = 4
    data_loader = DataLoader(src_dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=src_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        print('The code is run on GPU.')
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
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
    print('Start inference...')
    print(f'Total {len(data_loader)} samples to be evaluated.')
    frame_jsonl_path = os.path.join(opt.model_dir, "comm_metrics_frame.jsonl")
    epoch_csv_path = os.path.join(opt.model_dir, "comm_metrics_epoch.csv")
    comm_stats_list = []
    if not os.path.exists(epoch_csv_path):
        with open(epoch_csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch", "train_loss", "val_loss", "ap_50", "ap_70",
                "feature_bytes_per_frame", "metadata_bytes_per_frame", "total_bytes_per_frame",
                "normalized_ratio", "active_ratio", "active_neighbors_ratio",
                "packet_loss_rate", "fps"
            ])
    for i, batch_data in tqdm(enumerate(data_loader)):
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
                with open(frame_jsonl_path, "a") as f:
                    payload = {"frame_idx": int(i), **{k: float(v) for k, v in comm_stats.items()}}
                    f.write(json.dumps(payload) + "\n")

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

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)

    # Merge AP + communication summary
    eval_path = os.path.join(opt.model_dir, 'eval_global_sort.yaml' if opt.global_sort_detections else 'eval.yaml')
    summary = yaml_utils.load_yaml(eval_path) if os.path.exists(eval_path) else {}
    if len(comm_stats_list) > 0:
        def _avg(key, default=0.0):
            vals = [float(s.get(key, default)) for s in comm_stats_list]
            return float(sum(vals) / max(len(vals), 1))
        summary.update({
            "comm_feature_bytes_per_frame": _avg("feature_bytes_per_frame", _avg("bytes_per_frame", 0.0)),
            "comm_metadata_bytes_per_frame": _avg("metadata_bytes_per_frame", 0.0),
            "comm_total_bytes_per_frame": _avg("total_bytes_per_frame", _avg("bytes_per_frame", 0.0)),
            "comm_normalized_ratio": _avg("normalized_ratio", 1.0),
            "comm_active_ratio": _avg("active_ratio", 1.0),
            "comm_active_neighbors_ratio": _avg("active_neighbors_ratio", 1.0),
            "comm_packet_loss_rate": _avg("packet_loss_rate", 0.0),
        })
        # Backward compatibility
        summary["comm_bytes_per_frame"] = summary["comm_total_bytes_per_frame"]
        with open(epoch_csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "inference", None, None,
                summary.get('ap_50', None), summary.get('ap_70', None),
                summary["comm_feature_bytes_per_frame"],
                summary["comm_metadata_bytes_per_frame"],
                summary["comm_total_bytes_per_frame"],
                summary["comm_normalized_ratio"],
                summary["comm_active_ratio"],
                summary["comm_active_neighbors_ratio"],
                summary["comm_packet_loss_rate"],
                None
            ])
    yaml_utils.save_yaml(summary, os.path.join(opt.model_dir, "summary_eval.yaml"))

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
