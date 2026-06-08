import os
import sys
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter_input
from lib.data.datareader_h36m import DataReaderH36M  
from lib.model.loss import *
from collections import OrderedDict


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_file_logging(checkpoint_dir):
    log_path = os.path.join(checkpoint_dir, "train_console.log")
    log_file = open(log_path, "a", buffering=1)
    if not isinstance(sys.stdout, TeeLogger):
        sys.stdout = TeeLogger(sys.stdout, log_file)
        sys.stderr = TeeLogger(sys.stderr, log_file)
    print("INFO: Logging console output to", log_path)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate(args, model_pos, test_loader):
    print('INFO: Testing')
    device = get_device()
    results_all_rec = []
    results_all_mask = []
    results_all_noise = []
    gts_all = []
    args.aug = Augmenter_input(args)
    model_pos.eval()            
    with torch.no_grad():
        for batch_gt in tqdm(test_loader):
            batch_input_rec = batch_gt.clone()
            batch_input_mask = batch_gt.clone()
            batch_input_noise = batch_gt.clone()
            batch_input_mask = args.aug.augment(batch_input_mask, mask=True)
            batch_input_noise = args.aug.augment(batch_input_noise, noise=True)
            
            N, T = batch_gt.shape[:2]
            batch_input_rec = batch_input_rec.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_input_noise = batch_input_noise.to(device)
            batch_gt = batch_gt.to(device)
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
                batch_input_rec = batch_input_rec - batch_input_rec[:,:,0:1,:]
                batch_input_mask = batch_input_mask - batch_input_mask[:,:,0:1,:]
                batch_input_noise = batch_input_noise - batch_input_noise[:,:,0:1,:]
            predicted_3d_pos_rec = model_pos(batch_input_rec)
            predicted_3d_pos_mask = model_pos(batch_input_mask)
            predicted_3d_pos_noise = model_pos(batch_input_noise)
            if args.rootrel:
                predicted_3d_pos_rec[:,:,0,:] = 0     # [N,T,17,3]
                predicted_3d_pos_mask[:,:,0,:] = 0     # [N,T,17,3]
                predicted_3d_pos_noise[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0
            results_all_rec.append(predicted_3d_pos_rec.cpu().numpy())
            results_all_mask.append(predicted_3d_pos_mask.cpu().numpy())
            results_all_noise.append(predicted_3d_pos_noise.cpu().numpy())
            gts_all.append(batch_gt.cpu().numpy())
            
    num_test_frames = len(gts_all)
    
    e1_all_rec = np.zeros(num_test_frames)
    e1_all_mask = np.zeros(num_test_frames)
    e1_all_noise = np.zeros(num_test_frames)
  
    for i in range(num_test_frames):    
        pred_rec = results_all_rec[i] * 1000
        pred_mask = results_all_mask[i] * 1000
        pred_noise = results_all_noise[i] * 1000
        gt = gts_all[i] * 1000
        # Root-relative Errors
        pred_rec = pred_rec - pred_rec[:,0:1,:]
        pred_mask = pred_mask - pred_mask[:,0:1,:]
        pred_noise = pred_noise - pred_noise[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1_rec = mpjpe(pred_rec, gt)
        e1_all_rec[i] += np.mean(err1_rec)
        err1_mask = mpjpe(pred_mask, gt)
        e1_all_mask[i] += np.mean(err1_mask)
        err1_noise = mpjpe(pred_noise, gt)
        e1_all_noise[i] += np.mean(err1_noise)
    
    print('----------')
    print("reconstruction task")
    e1_rec = np.mean(e1_all_rec)
    print('Protocol #1 Error (MPJPE):', e1_rec, 'mm')
    
    final_result_mask = []
    print('----------')
    print("mask recovery task")
    e1_mask = np.mean(e1_all_mask)
    print('Protocol #1 Error (MPJPE):', e1_mask, 'mm')
    
    final_result_noise = []
    print('----------')
    print("noise recovery task")
    e1_noise = np.mean(e1_all_noise)
    print('Protocol #1 Error (MPJPE):', e1_noise, 'mm')
    return e1_rec, results_all_rec, e1_mask, results_all_mask, e1_noise, results_all_noise

      
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt, epoch=0, train_writer=None, global_step_start=0):
    model_pos.train()
    device = get_device()
    log_interval = getattr(args, 'log_interval', 200)
    for idx, (batch_gt) in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_input = batch_gt.clone()
        batch_size = len(batch_input)     
        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)
        with torch.no_grad():
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
                batch_input = batch_input - batch_input[:,:,0:1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos, predicted_feat = model_pos(batch_input, return_all=True)    # (N, T, 17, 3)
        
        optimizer.zero_grad()
        
        if args.loss_type == "part":
            loss_3d_pos = []
            loss_3d_scale = []
            loss_3d_velocity = []
            loss_trajectory_interpart = []
            loss_trajectory_intrapart = []
            p_start = 0 
            for p_i, p in enumerate(args.parts_len):
                p_end = p_start + p
                pre_part = predicted_3d_pos[:,:,p_start:p_end]
                gt_part = batch_gt[:,:,p_start:p_end]
                pre_trajectory = pre_part.mean(dim=2)
                gt_trajectory = gt_part.mean(dim=2)
                loss_3d_pos.append(loss_mpjpe(pre_part, gt_part))
                loss_3d_scale.append(n_mpjpe(pre_part, gt_part))
                loss_3d_velocity.append(loss_velocity(pre_part, gt_part))
                loss_trajectory_interpart.append(loss_mpjpe(pre_trajectory, gt_trajectory))
                p_start_ = 0
                for p_i_, p_ in enumerate(args.parts_len):
                    p_end_ = p_start_ + p_
                    pre_part_ = predicted_3d_pos[:,:,p_start_:p_end_]
                    gt_part_ = batch_gt[:,:,p_start_:p_end_]
                    pre_trajectory_ = pre_part_.mean(dim=2)
                    gt_trajectory_ = gt_part_.mean(dim=2)
                    loss_trajectory_intrapart.append(loss_mpjpe(pre_trajectory-pre_trajectory_, gt_trajectory-gt_trajectory_))
                    p_start_ = p_end_
                p_start = p_end
            loss_3d_pos = torch.stack(loss_3d_pos,dim=0)
            loss_3d_scale = torch.stack(loss_3d_scale,dim=0)
            loss_3d_velocity = torch.stack(loss_3d_velocity,dim=0)
            loss_trajectory_interpart = torch.stack(loss_trajectory_interpart,dim=0)
            loss_trajectory_intrapart = torch.stack(loss_trajectory_intrapart,dim=0)
            loss_total = loss_3d_pos.mean() + \
                        args.lambda_scale       * loss_3d_scale.mean() + \
                        args.lambda_3d_velocity * loss_3d_velocity.mean() + \
                        args.lambda_trajectory_interpart * loss_trajectory_interpart.mean() + \
                        args.lambda_trajectory_intrapart * loss_trajectory_intrapart.mean()
            losses['3d_pos'].update(loss_3d_pos.mean().item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.mean().item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.mean().item(), batch_size)
            losses['interpart_trajectory'].update(loss_trajectory_interpart.mean().item(), batch_size)
            losses['intrapart_trajectory'].update(loss_trajectory_intrapart.mean().item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity_feat = loss_velocity_feat(predicted_feat)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_total = loss_3d_pos + \
                        args.lambda_scale       * loss_3d_scale + \
                        args.lambda_3d_velocity_feat * loss_3d_velocity_feat + \
                        args.lambda_3d_velocity * loss_3d_velocity + \
                        args.lambda_lv          * loss_lv + \
                        args.lambda_lg          * loss_lg + \
                        args.lambda_a           * loss_a  + \
                        args.lambda_av          * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity_feat'].update(loss_3d_velocity_feat.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        
        loss_total.backward()
        optimizer.step()
        global_step = global_step_start + idx + 1
        if idx == 0 or global_step % log_interval == 0 or idx + 1 == len(train_loader):
            lr_now = optimizer.param_groups[0]['lr']
            print(
                '[epoch %d] iter %d/%d global_step %d lr %.8f total %.6f '
                '3d_pos %.6f 3d_scale %.6f 3d_velocity %.6f 3d_velocity_feat %.6f' % (
                    epoch + 1,
                    idx + 1,
                    len(train_loader),
                    global_step,
                    lr_now,
                    losses['total'].avg,
                    losses['3d_pos'].avg,
                    losses['3d_scale'].avg,
                    losses['3d_velocity'].avg,
                    losses['3d_velocity_feat'].avg,
                ),
                flush=True,
            )
            if train_writer is not None:
                train_writer.add_scalar('iter/lr', lr_now, global_step)
                for loss_name, meter in losses.items():
                    train_writer.add_scalar(f'iter/{loss_name}', meter.avg, global_step)
                if torch.cuda.is_available():
                    train_writer.add_scalar('iter/gpu_memory_allocated_gb', torch.cuda.memory_allocated() / 1024**3, global_step)
                    train_writer.add_scalar('iter/gpu_memory_reserved_gb', torch.cuda.memory_reserved() / 1024**3, global_step)


# get the subset
def get_fixed_subset(dataset, ratio=0.2, seed=42):
    np.random.seed(seed)
    total_size = len(dataset)
    subset_size = int(total_size * ratio)
    indices = np.random.choice(total_size, subset_size, replace=False)
    return Subset(dataset, indices)


def has_dynamic_anchors(args):
    return bool(getattr(args, 'anchor_subset_list', [])) and bool(getattr(args, 'anchor_sample_counts', {}))


def _stable_int(text):
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def sample_subset(dataset, sample_count, seed):
    total_size = len(dataset)
    sample_count = min(int(sample_count), total_size)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_size, sample_count, replace=False)
    return Subset(dataset, indices.tolist())


def build_dynamic_anchor_dataset(args, epoch):
    patient_dataset = MotionDataset3D(args, args.subset_list, "train")
    anchor_counts = dict(getattr(args, 'anchor_sample_counts', {}))
    anchor_seed = int(getattr(args, 'anchor_seed', 20260512))
    datasets = [patient_dataset]
    summary = [f"patient={len(patient_dataset)}"]

    for subset_name in getattr(args, 'anchor_subset_list', []):
        anchor_dataset = MotionDataset3D(args, [subset_name], "train")
        sample_count = anchor_counts.get(subset_name, anchor_counts.get(str(subset_name), len(anchor_dataset)))
        seed = anchor_seed + epoch * 1009 + _stable_int(str(subset_name))
        anchor_subset = sample_subset(anchor_dataset, sample_count, seed)
        datasets.append(anchor_subset)
        summary.append(f"{subset_name}={len(anchor_subset)}/{len(anchor_dataset)}")

    return ConcatDataset(datasets), ", ".join(summary)


def build_train_loader(args, trainloader_params, epoch=0):
    if has_dynamic_anchors(args):
        train_dataset, summary = build_dynamic_anchor_dataset(args, epoch)
    else:
        train_dataset = MotionDataset3D(args, args.subset_list, "train")
        train_dataset = get_fixed_subset(train_dataset, ratio=args.subset_ratio, seed=123)
        summary = f"fixed={len(train_dataset)}"
    return DataLoader(train_dataset, **trainloader_params), summary

def train_with_config(args, opts):
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    setup_file_logging(opts.checkpoint)
    print(args)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))


    print('Loading dataset...')
    num_workers = getattr(args, 'num_workers', 12)
    pin_memory = getattr(args, 'pin_memory', True)
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': num_workers,
          'pin_memory': pin_memory,
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': num_workers,
          'pin_memory': pin_memory,
    }
    if num_workers > 0:
        trainloader_params.update({
            'prefetch_factor': 4,
            'persistent_workers': True,
        })
        testloader_params.update({
            'prefetch_factor': 4,
            'persistent_workers': True,
        })

    test_dataset = MotionDataset3D(args, args.subset_list, "test")
    train_loader_3d, train_summary = build_train_loader(args, trainloader_params, epoch=0)
    test_loader = DataLoader(test_dataset, **testloader_params)
        
    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    device = get_device()
    print('INFO: Using device:', device)
    if device.type == 'cuda':
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.to(device)
    else:
        model_backbone = model_backbone.to(device)

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
      
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate: 
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)       
        st = 0
        lr_decay = args.lr_decay
        print('INFO: Training on {}(3D) batches ({})'.format(len(train_loader_3d), train_summary))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter_input(args)
        
        # Training
        for epoch in range(st, args.epochs):
            if has_dynamic_anchors(args):
                train_loader_3d, train_summary = build_train_loader(args, trainloader_params, epoch=epoch)
                print('INFO: Epoch {} training on {}(3D) batches ({})'.format(epoch + 1, len(train_loader_3d), train_summary))
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['3d_velocity_feat'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['interpart_trajectory'] = AverageMeter()
            losses['intrapart_trajectory'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            losses['total'] = AverageMeter()
            N = 0
            
            train_epoch(
                args,
                model_pos,
                train_loader_3d,
                losses,
                optimizer,
                has_3d=True,
                has_gt=True,
                epoch=epoch,
                train_writer=train_writer,
                global_step_start=epoch * len(train_loader_3d),
            )
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg))
                for loss_name, meter in losses.items():
                    train_writer.add_scalar(f'epoch/{loss_name}', meter.avg, epoch + 1)
                train_writer.add_scalar('epoch/lr', lr, epoch + 1)
                train_writer.add_scalar('epoch/time_min', elapsed, epoch + 1)
            else:
                e1_rec, results_all_rec, e1_mask, results_all_mask, e1_noise, results_all_noise = evaluate(args, model_pos, test_loader)
                print('[%d] time %.2f lr %f 3d_train %f e1_rec %f  e1_mask %f e1_noise  %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1_rec,
                    e1_mask,
                    e1_noise))
                train_writer.add_scalar('Error P1 of reconstruction', e1_rec, epoch + 1)
                train_writer.add_scalar('Error P1 of mask recovery', e1_mask, epoch + 1)
                train_writer.add_scalar('Error P1 of noise recovery', e1_noise, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity_feat', losses['3d_velocity_feat'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_interpart_trajectory', losses['interpart_trajectory'].avg, epoch + 1)
                train_writer.add_scalar('loss_intrapart_trajectory', losses['intrapart_trajectory'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if not args.no_eval and e1_rec < min_loss:
                min_loss = e1_rec
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
                
    if opts.evaluate:
        e1_rec, results_all_rec, e1_mask, results_all_mask, e1_noise, results_all_noise = evaluate(args, model_pos, test_loader)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    if opts.checkpoint == "checkpoint":
        opts.checkpoint = "checkpoint/motionbert"
    train_with_config(args, opts)
