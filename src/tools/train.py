# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib




import argparse
import os
import statistics
import json
import csv
import hashlib
from datetime import datetime
import torch
import tqdm
import platform
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import src.hypes_yaml.yaml_utils as yaml_utils
from src.tools import train_utils
from src.tools import multi_gpu_utils
from src.data_utils.datasets import build_dataset
from src.tools import train_utils
import warnings
import time
import numpy as np
warnings.filterwarnings("ignore")


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _append_jsonl(path, payload):
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _safe_float(x, default=0.0):
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float(default)


def _latest_checkpoint_name(folder):
    ckpts = [f for f in os.listdir(folder) if f.startswith("net_epoch") and f.endswith(".pth")]
    if len(ckpts) == 0:
        return None
    ckpts.sort(key=lambda n: int("".join([c for c in n if c.isdigit()]) or 0))
    return ckpts[-1]


def _config_fingerprint(hypes):
    return hashlib.sha256(str(hypes).encode("utf-8")).hexdigest()


def compute_comm_losses(hypes, output_dict, device):
    comm_cfg = hypes.get('communication', {})
    stats = output_dict.get('comm_stats', {})
    aux = output_dict.get('comm_aux', {})
    losses = {}
    total_aux = torch.tensor(0.0, device=device)

    learn_cfg = comm_cfg.get('learnable_mask', {})
    if bool(comm_cfg.get('enabled', False)) and bool(learn_cfg.get('enabled', False)):
        lam_sparse = float(learn_cfg.get('sparsity_lambda', 0.0))
        target_ratio = float(learn_cfg.get('target_ratio', 0.10))
        lam_budget = float(learn_cfg.get('budget_lambda', 0.0))
        use_budget = bool(learn_cfg.get('use_budget_loss', False))
        mask_mean = aux.get('mask_mean', None)
        if mask_mean is not None:
            if lam_sparse > 0:
                l_sparse = lam_sparse * mask_mean
                losses['sparse_loss'] = float(l_sparse.detach().cpu().item())
                # Backward-compatible name
                losses['comm_loss'] = losses['sparse_loss']
                total_aux = total_aux + l_sparse
            if use_budget and lam_budget > 0:
                l_budget = lam_budget * torch.relu(mask_mean - target_ratio) ** 2
                losses['budget_loss'] = float(l_budget.detach().cpu().item())
                total_aux = total_aux + l_budget

    repair_cfg = comm_cfg.get('repair_network', {})
    if bool(comm_cfg.get('enabled', False)) and bool(repair_cfg.get('enabled', False)):
        pred = aux.get('repair_pred', None)
        target = aux.get('repair_target', None)
        w = float(repair_cfg.get('loss_weight', 0.0))
        if pred is not None and target is not None and w > 0:
            l_rep = w * torch.nn.functional.mse_loss(pred, target)
            losses['repair_loss'] = float(l_rep.detach().cpu().item())
            total_aux = total_aux + l_rep

    losses['total_aux_loss'] = float(total_aux.detach().cpu().item())
    return total_aux, losses, stats




def train_parser():

    

    """
    Initializes and returns a command-line argument parser for the training process.

    Usage examples:
        python train.py --hypes_yaml [path_to_yaml]
        python train.py --hypes_yaml [path_to_yaml] --half
    
    Resturn:
        opt: It will be something like 
        Namespace(dist_url='env://', half=False, hypes_yaml='src/hypes_yaml/point_pillar_intermediate_V2VAM.yaml', model_dir='')
    
    done
    """
    # Create an argument parser with a brief description
    parser = argparse.ArgumentParser(description="Parser for training with synthetic data")

    # Required: Path to the YAML configuration file for hyperparameters
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help="Path to the YAML configuration file for training")

    # Optional: Directory for saving or loading model checkpoints
    parser.add_argument('--model_dir', default='',
                        help="Directory to load/save the model checkpoint")

    # Optional: Enable mixed-precision training (half-precision)
    parser.add_argument("--half", action='store_true',
                        help="Use half-precision training to reduce memory consumption")

    # Optional: Distributed training initialization URL (default uses environment variable)
    parser.add_argument('--dist_url', default='env://',
                        help="URL for initializing distributed training")

    # Parse and return the command-line arguments
    opt = parser.parse_args()
    return opt


start_time = time.time()
def main():
    num_workers = 4
    prefetch_factor = 4
    opt = train_parser()
    print("*********************Step0: Train parser completed *********************")
    print('You passed the following options:\n',opt)
    
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print("*********************Step1: Yaml file Readed*********************")
    print("The configuration setup read from above path is as follow:\n",hypes)

    
    print('*********************Step2: Multi GPU Checking*********************',end="\n")
    multi_gpu_utils.init_distributed_mode(opt)
    

    print('*********************Step3_1: Train Dataset Building*********************')
    src_train_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=True)
    print('*********************Step3_2: Validate Dataset Building*********************')
    src_validate_dataset = build_dataset(dataset_cfg=hypes, visualize=False, train=False)
    
    if opt.distributed:
        print('*********************Step4: DataLoader Creating*********************')
        print('Since Distributed training environment detected. Initializing DistributedSampler for data parallelization is done.')
        sampler_train = DistributedSampler(src_train_dataset)
        sampler_val = DistributedSampler(src_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        print(f"Numnber of workers was set to:{num_workers}")
        train_loader = DataLoader(src_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=num_workers, # These were 8, due to error I set them to 0
                                  collate_fn=src_train_dataset.collate_batch_train)
        val_loader = DataLoader(src_validate_dataset,
                                sampler=sampler_val,
                                num_workers=num_workers,
                                collate_fn=src_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        print('*********************Step4: DataLoader Creating*********************')
        print('Since Distributed training environment not detected. Initializing DataLoader for data parallelization is done.')
        print(f"Numnber of workers was set to:{num_workers}")
        train_loader = DataLoader(src_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=num_workers,
                                  collate_fn=src_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  prefetch_factor=prefetch_factor)
        val_loader = DataLoader(src_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=num_workers,
                                collate_fn=src_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=prefetch_factor)

    print('*********************Step5: Creating Model*********************')
    model = train_utils.create_model(hypes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will run ({device}) device.')
    # if we want to train from last checkpoint.
    if opt.model_dir:
        print(f'Model directory {opt.model_dir} detected. Initializing model from saved checkpoint.')
        saved_path = opt.model_dir
        init_epoch = 0
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    comm_logging_cfg = hypes.get("communication", {}).get("logging", {})
    save_csv = bool(comm_logging_cfg.get("save_csv", True))
    save_step_json = bool(comm_logging_cfg.get("save_per_step_json", True))
    save_epoch_json = bool(comm_logging_cfg.get("save_per_epoch_json", True))
    save_validation_json = bool(comm_logging_cfg.get("save_validation_json", True))
    run_info_path = os.path.join(saved_path, "run_info.json")
    event_log_path = os.path.join(saved_path, "pipeline_events.log")
    train_step_jsonl = os.path.join(saved_path, "train_step_metrics.jsonl")
    val_step_jsonl = os.path.join(saved_path, "val_step_metrics.jsonl")
    epoch_jsonl = os.path.join(saved_path, "train_epoch_metrics.jsonl")
    comm_csv_path = os.path.join(saved_path, "comm_metrics_epoch.csv")

    def _event(msg):
        line = f"[{_utc_now()}] {msg}"
        print(line)
        with open(event_log_path, "a") as f:
            f.write(line + "\n")

    run_info = {
        "mode": "train",
        "created_at_utc": _utc_now(),
        "hypes_yaml": opt.hypes_yaml,
        "model_dir_arg": opt.model_dir,
        "saved_path": saved_path,
        "distributed": bool(getattr(opt, "distributed", False)),
        "half_precision": bool(opt.half),
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset_train_size": len(src_train_dataset),
        "dataset_validate_size": len(src_validate_dataset),
        "batch_size": int(hypes["train_params"]["batch_size"]),
        "epoches_config": int(hypes["train_params"]["epoches"]),
        "config_fingerprint_sha256": _config_fingerprint(hypes),
        "checkpoint_on_start": _latest_checkpoint_name(saved_path) if os.path.isdir(saved_path) else None,
    }
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    _event(f"Run initialized. saved_path={saved_path}")
    _event(f"Config fingerprint={run_info['config_fingerprint_sha256']}")
    _event(f"Dataset sizes: train={len(src_train_dataset)} validate={len(src_validate_dataset)}")

    if save_csv and not os.path.exists(comm_csv_path):
        with open(comm_csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch", "train_loss", "val_loss", "ap_50", "ap_70",
                "feature_bytes_per_frame", "context_bytes_per_frame", "metadata_bytes_per_frame", "total_bytes_per_frame",
                "normalized_ratio", "active_ratio", "active_neighbors_ratio",
                "packet_loss_rate", "receiver_request_keep_ratio",
                "receiver_request_context_ratio", "receiver_request_mask_metadata_ratio",
                "fps"
            ])

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    print('*********************Step6: Creating Loss Function*********************')
    criterion = train_utils.create_loss(hypes)
    print("Conf Loss = Confidence loss = Classification loss * Classification_weight")
    print("Loc Loss = reg_loss")
    print("Total loss = reg_loss + conf_loss")

    # optimizer setup
    print('*********************Step7: Creating Optimizer*********************')
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if opt.model_dir:
        init_epoch, model_without_ddp, optimizer, scheduler, scaler = \
            train_utils.load_training_state(saved_path,
                                           model_without_ddp,
                                           optimizer,
                                           scheduler,
                                           scaler)
        _event(f"Resumed training state from {saved_path}, init_epoch={init_epoch}")
    else:
        _event("Starting training from scratch (new run directory).")

    print('*********************Step8: Starting Training part*********************')
    epoches = hypes['train_params']['epoches']
    batch_size = hypes['train_params']['batch_size']
    # used to help schedule learning rate
    print(f"Batch_size = {batch_size}")
    print(f"Number of epochs = {epoches}")
    print(f"Number of batch_Data to analyse in each epoch = {len(train_loader)}")
    _event(f"Training plan: start_epoch={init_epoch}, end_epoch={max(epoches, init_epoch)-1}, batches_per_epoch={len(train_loader)}")

    
    loss_dict = {}
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        st = time.time()

        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('-'*25)
            print()
            print(f"For Epoch {epoch}:")
            print('learning rate is %.7f ' % param_group["lr"])
            print(' || ')
            print(" || ")
            print(' ** ')
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        writer.add_scalar("LR/epoch", param_group["lr"], epoch)
        writer.flush()
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        

        index =0
        train_loss_batch=[]
        train_comm_stats = []
        print("Training Epoch %d .........." % epoch)   
        for batch_data in train_loader:
            step_st = time.time()

            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
                aux_total, aux_breakdown, comm_stats = compute_comm_losses(hypes, ouput_dict, device)
                final_loss = final_loss + aux_total
                criterion.loss_dict['total_loss'] = final_loss
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    # if epoch == 0:
                    #     make_dot(ouput_dict, params=dict(model.named_parameters())).render("model_graph", format="png")
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    aux_total, aux_breakdown, comm_stats = compute_comm_losses(hypes, ouput_dict, device)
                    final_loss = final_loss + aux_total
                    criterion.loss_dict['total_loss'] = final_loss

            
            criterion.logging(epoch, index, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
       
            loss_value = final_loss.item()


            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + index)
            train_loss_batch.append(loss_value)
            writer.add_scalar('Train_Loss/batch', loss_value,
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/bytes_per_frame',
                              float(comm_stats.get('bytes_per_frame', 0.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/feature_bytes_per_frame',
                              float(comm_stats.get('feature_bytes_per_frame', comm_stats.get('bytes_per_frame', 0.0))),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/total_bytes_per_frame',
                              float(comm_stats.get('total_bytes_per_frame', comm_stats.get('bytes_per_frame', 0.0))),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/context_bytes_per_frame',
                              float(comm_stats.get('context_bytes_per_frame', 0.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/normalized_ratio',
                              float(comm_stats.get('normalized_ratio', 1.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/active_ratio',
                              float(comm_stats.get('active_ratio', 1.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/active_neighbors',
                              float(comm_stats.get('active_neighbors_ratio', 1.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/loss_rate',
                              float(comm_stats.get('packet_loss_rate', 0.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/aux_total_loss',
                              float(aux_breakdown.get('total_aux_loss', 0.0)),
                              epoch * len(train_loader) + index)
            mask_mean = ouput_dict.get('comm_aux', {}).get('mask_mean', None)
            if mask_mean is not None:
                writer.add_scalar('Comm/mask_mean',
                                  float(mask_mean.detach().cpu().item()),
                                  epoch * len(train_loader) + index)
            writer.add_scalar('Comm/L_sparse',
                              float(aux_breakdown.get('sparse_loss', aux_breakdown.get('comm_loss', 0.0))),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Comm/L_budget',
                              float(aux_breakdown.get('budget_loss', 0.0)),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Loss/L_det',
                              float((final_loss - aux_total).detach().cpu().item()),
                              epoch * len(train_loader) + index)
            writer.add_scalar('Loss/total',
                              float(final_loss.detach().cpu().item()),
                              epoch * len(train_loader) + index)
            writer.flush()

            if save_step_json:
                _append_jsonl(train_step_jsonl, {
                    "ts_utc": _utc_now(),
                    "epoch": int(epoch),
                    "batch_index": int(index),
                    "global_step": int(epoch * len(train_loader) + index),
                    "lr": _safe_float(param_group["lr"]),
                    "loss_total": _safe_float(final_loss.detach().cpu().item()),
                    "loss_det": _safe_float((final_loss - aux_total).detach().cpu().item()),
                    "loss_aux_total": _safe_float(aux_breakdown.get("total_aux_loss", 0.0)),
                    "loss_sparse": _safe_float(aux_breakdown.get("sparse_loss", aux_breakdown.get("comm_loss", 0.0))),
                    "loss_budget": _safe_float(aux_breakdown.get("budget_loss", 0.0)),
                    "loss_repair": _safe_float(aux_breakdown.get("repair_loss", 0.0)),
                    "mask_mean": _safe_float(ouput_dict.get('comm_aux', {}).get('mask_mean', 0.0)),
                    "comm_stats": {k: _safe_float(v) for k, v in comm_stats.items()},
                    "step_seconds": _safe_float(time.time() - step_st),
                })
            index +=1
            train_comm_stats.append(comm_stats)


        print(f"\nAt epoch {epoch}, the mean train loss is: {np.mean(train_loss_batch)}")
        print(f"At epoch {epoch}, the min train loss is: {np.min(train_loss_batch)}")
        
        writer.add_scalar('Train_Loss/epoch', np.mean(train_loss_batch), epoch)
        writer.flush()

   

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
            train_utils.save_training_state(saved_path,
                                           epoch,
                                           model_without_ddp,
                                           optimizer,
                                           scheduler,
                                           scaler)
            _event(f"Checkpoint saved: net_epoch{epoch + 1}.pth")

        if epoch % hypes['train_params']['eval_freq'] == 0:
            

            with torch.no_grad():
                valid_loss_batch = []
                val_comm_stats = []
                print('Validating epoch %d ..........' % epoch)
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    aux_total, aux_breakdown, comm_stats = compute_comm_losses(hypes, ouput_dict, device)
                    final_loss = final_loss + aux_total
                    valid_loss_batch.append(final_loss.item())
                    val_comm_stats.append(comm_stats)
                    writer.add_scalar('Validate_Loss/batch',
                                      final_loss.item(),
                                      epoch * len(val_loader) + i)
                    writer.flush()
                    if save_validation_json:
                        _append_jsonl(val_step_jsonl, {
                            "ts_utc": _utc_now(),
                            "epoch": int(epoch),
                            "batch_index": int(i),
                            "global_step": int(epoch * len(val_loader) + i),
                            "loss_total": _safe_float(final_loss.detach().cpu().item()),
                            "loss_aux_total": _safe_float(aux_breakdown.get("total_aux_loss", 0.0)),
                            "comm_stats": {k: _safe_float(v) for k, v in comm_stats.items()},
                        })

            print('At epoch %d, the mean validation loss is: %f' % (epoch,
                                                              np.mean(valid_loss_batch)))
            print('At epoch %d, the min validation loss is: %f' % (epoch,
                                                              np.min(valid_loss_batch)))
            writer.add_scalar('Validate_Loss/epoch', np.mean(valid_loss_batch), epoch)
            writer.flush()

        loss_dict[epoch] = {
            'train_loss': np.mean(train_loss_batch),
            'train_loss_min': np.min(train_loss_batch),
            'train_loss_batch': train_loss_batch,
            'val_loss': np.mean(valid_loss_batch) if 'valid_loss_batch' in locals() else None,
            'val_loss_min': np.min(valid_loss_batch) if 'valid_loss_batch' in locals() else None,
            'val_loss_batch': valid_loss_batch if 'valid_loss_batch' in locals() else None
        }
        with open(os.path.join(saved_path, f'loss_dict_epoch_{epoch}.json'), 'w') as f:
            json.dump(loss_dict, f, indent=4)

        stat_src = val_comm_stats if 'val_comm_stats' in locals() and len(val_comm_stats) > 0 else train_comm_stats
        if len(stat_src) > 0:
            feature_bytes_pf = float(np.mean([s.get('feature_bytes_per_frame', s.get('bytes_per_frame', 0.0)) for s in stat_src]))
            context_bytes_pf = float(np.mean([s.get('context_bytes_per_frame', 0.0) for s in stat_src]))
            metadata_bytes_pf = float(np.mean([s.get('metadata_bytes_per_frame', 0.0) for s in stat_src]))
            total_bytes_pf = float(np.mean([s.get('total_bytes_per_frame', s.get('bytes_per_frame', 0.0)) for s in stat_src]))
            normalized_ratio = float(np.mean([s.get('normalized_ratio', 1.0) for s in stat_src]))
            active_ratio = float(np.mean([s.get('active_ratio', 1.0) for s in stat_src]))
            active_neighbors = float(np.mean([s.get('active_neighbors_ratio', 1.0) for s in stat_src]))
            loss_rate = float(np.mean([s.get('packet_loss_rate', 0.0) for s in stat_src]))
            rr_keep_ratio = float(np.mean([s.get('receiver_request_keep_ratio', 1.0) for s in stat_src]))
            rr_ctx_ratio = float(np.mean([s.get('receiver_request_context_ratio', 0.0) for s in stat_src]))
            rr_meta_ratio = float(np.mean([s.get('receiver_request_mask_metadata_ratio', 0.0) for s in stat_src]))
        else:
            feature_bytes_pf, context_bytes_pf, metadata_bytes_pf, total_bytes_pf, normalized_ratio, active_ratio, active_neighbors, loss_rate, rr_keep_ratio, rr_ctx_ratio, rr_meta_ratio = 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0

        sp = time.time()
        epoch_sec = max(sp - st, 1e-6)
        fps = float(len(train_loader) * batch_size / epoch_sec)
        writer.add_scalar('Comm/fps', fps, epoch)
        writer.flush()
        if save_csv:
            with open(comm_csv_path, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    epoch, float(np.mean(train_loss_batch)),
                    float(np.mean(valid_loss_batch)) if 'valid_loss_batch' in locals() else None,
                    None, None,
                    feature_bytes_pf, context_bytes_pf, metadata_bytes_pf, total_bytes_pf, normalized_ratio,
                    active_ratio, active_neighbors, loss_rate, rr_keep_ratio, rr_ctx_ratio, rr_meta_ratio, fps
                ])

        if save_epoch_json:
            _append_jsonl(epoch_jsonl, {
                "ts_utc": _utc_now(),
                "epoch": int(epoch),
                "train_loss_mean": _safe_float(np.mean(train_loss_batch)),
                "train_loss_min": _safe_float(np.min(train_loss_batch)),
                "val_loss_mean": _safe_float(np.mean(valid_loss_batch)) if 'valid_loss_batch' in locals() else None,
                "val_loss_min": _safe_float(np.min(valid_loss_batch)) if 'valid_loss_batch' in locals() else None,
                "fps": _safe_float(fps),
                "epoch_seconds": _safe_float(epoch_sec),
                "comm_feature_bytes_per_frame": _safe_float(feature_bytes_pf),
                "comm_context_bytes_per_frame": _safe_float(context_bytes_pf),
                "comm_metadata_bytes_per_frame": _safe_float(metadata_bytes_pf),
                "comm_total_bytes_per_frame": _safe_float(total_bytes_pf),
                "comm_normalized_ratio": _safe_float(normalized_ratio),
                "comm_active_ratio": _safe_float(active_ratio),
                "comm_active_neighbors_ratio": _safe_float(active_neighbors),
                "comm_packet_loss_rate": _safe_float(loss_rate),
                "receiver_request_keep_ratio": _safe_float(rr_keep_ratio),
                "receiver_request_context_ratio": _safe_float(rr_ctx_ratio),
                "receiver_request_mask_metadata_ratio": _safe_float(rr_meta_ratio),
            })

        _event(
            f"Epoch {epoch} done | train_loss={np.mean(train_loss_batch):.6f} "
            f"val_loss={(np.mean(valid_loss_batch) if 'valid_loss_batch' in locals() else float('nan')):.6f} "
            f"fps={fps:.3f} comm_norm={normalized_ratio:.4f}"
        )

        print(f"Total training time for epoch {epoch} : {((sp - st)/60)} minutes")
        print('-'*50)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    _event("Training finished successfully.")
    run_info["completed_at_utc"] = _utc_now()
    run_info["completed"] = True
    run_info["final_checkpoint"] = _latest_checkpoint_name(saved_path)
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)


if __name__ == '__main__':
    main()
end_time = time.time()
print(f"Total training time: {((end_time - start_time)/60)} minutes")
