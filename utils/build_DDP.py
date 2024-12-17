import os
from torch.nn import ModuleList, parallel
import torch.distributed as dist
from torch.backends import cudnn
import torch.multiprocessing as mp
from torch.cuda import set_device

def DDP(net, rank):
    cls_net = parallel.DistributedDataParallel(net[0], device_ids=[rank])
    spl_net = parallel.DistributedDataParallel(net[1], device_ids=[rank])
    net = ModuleList([cls_net, spl_net])
    return net

def run(main, config):
    cudnn.benchmark = True
    mp.spawn(main, args=(config,), nprocs=config['world_size'], join=True)
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    dist.barrier()
    set_device(rank)

def cleanup():
    dist.destroy_process_group()