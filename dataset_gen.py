#!/usr/bin/env python

import os
import argparse
import yaml
from utils.data_preprocess import create_mimic, create_chexpert, create_chestxray8, create_openi
from utils.logger import create_logger

def main(args):
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    logger = create_logger(output_dir=config['log_path'], name='Generating datasets')
    if args.test_set:
        test_target = os.path.join(config['data_path'], args.test_set)
        if args.test_set == 'chexpert':
            create_chexpert(args.data_root, test_target, logger)
        elif args.test_set == 'chestxray8':
            create_chestxray8(args.data_root, test_target, logger)
        elif args.test_set == 'openi':
            create_openi(args.data_root, test_target, logger)
    else:
        train_target = os.path.join(config['data_path'], config['train_set'])
        if config['train_set'] == 'mimic':
            create_mimic(args.data_root, train_target, logger)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/swin384.yaml', help='load the config file')
    parser.add_argument('--test_set', type=str, default='', help='test set name, e.g., chexpert, chestxray8, openi. Leave it blank means generate train set. You can customize your own data generator')
    parser.add_argument('--data_root', type=str, default='/data', help='the path where your raw dataset located')
    args = parser.parse_args()
    main(args)