#!/usr/bin/env python

import os
import yaml
import torch
import argparse
import torch.nn.functional as F
from multiprocessing import cpu_count
from torch.cuda import device_count
from torch.utils.tensorboard import SummaryWriter
from utils.build_data import build_trainloader, build_envloader
from utils.build_net import build_all_net
from utils.logger import create_logger
from utils.get_weights import load_pretrained, load_checkpoint, save_checkpoint
from utils.get_loss_fn import irm_loss, nce_loss
from utils.optimizer import build_all_opt
from utils.lr_scheduler import build_all_lrs
from utils.scaler import build_all_scaler
from utils.build_DDP import DDP, run, setup, cleanup
from utils.train_module import train_cls_module, train_spl_module, validate_module

def main(rank, config):
    setup(rank, config['world_size'])
    logger = create_logger(output_dir=config['log_path'], dist_rank=rank, name='Training')
    init_trainloader = build_trainloader(rank, config)
    train_loader = init_trainloader.train_loader()
    val_loader = init_trainloader.val_loader()
    net = build_all_net(config, init_trainloader) 
    net.cuda(rank)
    net_without_ddp = net
    net = DDP(net, rank)
    opt = build_all_opt(net)
    lrs = build_all_lrs(opt)
    scaler = build_all_scaler()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if config['resume_epoch']:
        load_checkpoint(config, net_without_ddp, opt, lrs, scaler, map_location, logger)
    else:
        load_pretrained(config, net_without_ddp[0], torch.device('cuda:%d' % rank), logger)    
    soft_split = torch.randn(init_trainloader.num_train, config['n_env']).cuda(rank)
    split_all = F.one_hot(soft_split.argmax(dim=1))
    irm_lossfn = irm_loss(init_trainloader.num_classes)
    nce_lossfn = nce_loss()
    logger.info('Start Tensorboard with "tensorboard --logdir=logs"')
    writer = SummaryWriter(log_dir=config['log_path'])
    max_accuracy = 0
    best_epoch = 0
    for epoch in range(100):
        train_loader.sampler.set_epoch(epoch)
        logger.info('=' * 50)
        logger.info(f'Epoch {epoch} begin training and environment splitting......')
        conf, h, idx = train_cls_module(config, rank, epoch, net[0], split_all, irm_lossfn, train_loader, opt[0], lrs[0], scaler[0], logger, writer)
        env_loader = build_envloader(config, conf, h, idx)
        split_all = train_spl_module(config, rank, epoch, net[1], nce_lossfn, env_loader, opt[1], lrs[1], scaler[1], logger, writer)
        logger.info(f'Epoch {epoch} training and environment splitting finished !!!')
        acc_or_auc = validate_module(config, rank, epoch, net[0], val_loader, logger, writer)
        is_best = acc_or_auc > max_accuracy
        if rank == 0:
            save_checkpoint(epoch, config, net_without_ddp, opt, lrs, scaler, is_best, logger)
        if is_best:
            max_accuracy = acc_or_auc
            best_epoch = epoch
        if init_trainloader.num_classes == 1:
            logger.info(f'The best epoch is {best_epoch}, the max AUC is {max_accuracy:.3f}')
        else:
            logger.info(f'The best epoch is {best_epoch}, the max Acc is {max_accuracy:.3f}')
    writer.close()
    cleanup()   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/swin384.yaml', help='load the config file')
    parser.add_argument('--local_rank', type=str, default=None, help='GPU ids, if multiple devices, please separate by comma, default None means all devices')
    parser.add_argument('--cpu_count', type=int, default=None, help='CPU counts, default None means all CPUs')
    args = parser.parse_args()
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    if args.local_rank is not None:        
        os.environ['CUDA_VISIBLE_DEVICES'] = args.local_rank
    n_gpus = device_count()
    if args.cpu_count is None:
        n_cpus = cpu_count()
    else:
        n_cpus = args.cpu_count
    omp_num_threads = n_cpus // n_gpus
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    config.update({'num_workers': omp_num_threads, 'world_size': n_gpus})
    run(main, config)