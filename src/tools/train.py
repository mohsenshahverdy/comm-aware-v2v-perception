# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib




import argparse
import os
import statistics
import json
import csv
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


def compute_comm_losses(hypes, output_dict, device):
    comm_cfg = hypes.get('communication', {})
    stats = output_dict.get('comm_stats', {})
    aux = output_dict.get('comm_aux', {})
    losses = {}
    total_aux = torch.tensor(0.0, device=device)

    learn_cfg = comm_cfg.get('learnable_mask', {})
    if bool(comm_cfg.get('enabled', False)) and bool(learn_cfg.get('enabled', False)):
        lam = float(learn_cfg.get('sparsity_lambda', 0.0))
        mask_mean = aux.get('mask_mean', None)
        if mask_mean is not None and lam > 0:
            l_comm = lam * mask_mean
            losses['comm_loss'] = float(l_comm.detach().cpu().item())
            total_aux = total_aux + l_comm

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

    comm_csv_path = os.path.join(saved_path, "comm_metrics_epoch.csv")
    if not os.path.exists(comm_csv_path):
        with open(comm_csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch", "train_loss", "val_loss", "ap_50", "ap_70",
                "bytes_per_frame", "active_ratio", "active_neighbors_ratio",
                "packet_loss_rate", "fps"
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

    print('*********************Step8: Starting Training part*********************')
    epoches = hypes['train_params']['epoches']
    batch_size = hypes['train_params']['batch_size']
    # used to help schedule learning rate
    print(f"Batch_size = {batch_size}")
    print(f"Number of epochs = {epoches}")
    print(f"Number of batch_Data to analyse in each epoch = {len(train_loader)}")

    
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
            writer.flush()
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
            bytes_pf = float(np.mean([s.get('bytes_per_frame', 0.0) for s in stat_src]))
            active_ratio = float(np.mean([s.get('active_ratio', 1.0) for s in stat_src]))
            active_neighbors = float(np.mean([s.get('active_neighbors_ratio', 1.0) for s in stat_src]))
            loss_rate = float(np.mean([s.get('packet_loss_rate', 0.0) for s in stat_src]))
        else:
            bytes_pf, active_ratio, active_neighbors, loss_rate = 0.0, 1.0, 1.0, 0.0

        sp = time.time()
        epoch_sec = max(sp - st, 1e-6)
        fps = float(len(train_loader) * batch_size / epoch_sec)
        writer.add_scalar('Comm/fps', fps, epoch)
        writer.flush()
        with open(comm_csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch, float(np.mean(train_loss_batch)),
                float(np.mean(valid_loss_batch)) if 'valid_loss_batch' in locals() else None,
                None, None,
                bytes_pf, active_ratio, active_neighbors, loss_rate, fps
            ])

        print(f"Total training time for epoch {epoch} : {((sp - st)/60)} minutes")
        print('-'*50)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
end_time = time.time()
print(f"Total training time: {((end_time - start_time)/60)} minutes")
