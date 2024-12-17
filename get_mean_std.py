#!/usr/bin/env python

import argparse
import yaml
import os
import numpy as np
from multiprocessing import cpu_count
from utils.mean_std import get_train_mean_std, get_test_mean_std
from utils.logger import create_logger

def main(args):
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    if args.cpu_count is None:
        n_cpus = cpu_count()
    else:
        n_cpus = args.cpu_count
    logger = create_logger(output_dir=config['log_path'], name='Mean-std')
    logger.info('Begin calculating mean and standard deviation......')
    if args.test_path:
        mean, std = get_test_mean_std(args.test_path, n_cpus)
    else:
        train_path = os.path.join(config['data_path'], config['train_set'])
        mean, std = get_train_mean_std(train_path, n_cpus)    
        np.savetxt('./temp/mean', mean)
        np.savetxt('./temp/std', std)
    logger.info('Calculating done !!!')
    logger.info(f'Mean is {["%.3f" % x for x in mean]}, std is {["%.3f" % x for x in std]}')
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/swin384.yaml', help='load the config file')
    parser.add_argument('--test_path', type=str, default='', help='if the dataset is a testset, please input test set path')
    parser.add_argument('--cpu_count', type=int, default=None, help='CPU counts, default None means all CPUs')
    args = parser.parse_args()
    main(args)