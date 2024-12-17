#!/usr/bin/env python

import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import cpu_count
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, average_precision_score, precision_score
from utils.build_data import build_testloader
from utils.build_net import build_swinv2
from utils.logger import create_logger
from utils.get_weights import load_weights
from utils.mean_std import get_test_mean_std

def main(args):
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    if args.cpu_count is None:
        n_cpus = cpu_count()
    else:
        n_cpus = args.cpu_count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = create_logger(output_dir=config['log_path'], name='Testing')
    if args.mean_std:
        mean, std = args.mean_std
        mean, std = [mean], [std]
    else:
        logger.info('Analysing images......')
        mean, std = get_test_mean_std(args.test_path, n_cpus)
        logger.info('Analysing done !!!')
    init_testloader = build_testloader(config, mean[0], std[0], args.batch_size)
    test_loader = init_testloader.test_loader(args.test_path, n_cpus)
    cls_net = build_swinv2(config, init_testloader.num_classes)
    cls_net.to(device)
    load_weights(args.ckpt_file, cls_net, device, logger)
    net = cls_net[0]
    logger.info('Begin testing......')
    net.eval()
    with torch.no_grad():
        y_all = torch.tensor([]).to(device)
        path_all = []
        for img, path_img in tqdm(test_loader, desc='Testing', dynamic_ncols=True):
            img = img.to(device)
            y, _, _ = net(img)
            y_all = torch.cat((y_all, y), 0)
            path_all.extend(path_img)
    logger.info('Testing finished !!!')
    label_map = np.loadtxt('./temp/label_map', dtype=str).tolist()
    if init_testloader.num_classes == 1:
        prob = torch.sigmoid(y_all).cpu().numpy().flatten()
        if args.label_file:
            label_raw = pd.read_csv(args.label_file, header=None, index_col=0, dtype=str).squeeze().to_dict()
            label = [label_map.index(label_raw[x]) for x in path_all]
            fpr, tpr, thresholds = roc_curve(label, prob)
            auroc = auc(fpr, tpr)
            pr = average_precision_score(label, prob)
            threshold = thresholds[np.argmax(tpr - fpr)]
            np.savetxt('./temp/threshold', [threshold])
            predict = np.where(prob > threshold, 1, 0)
            f1 = f1_score(label, predict)
            recall = recall_score(label, predict)
            precision = precision_score(label, predict)
            logger.info(f'AUC is {auroc:.3f}, PR is {pr:.3f}, recall is {recall:.3f}, precision is {precision:.3f}, F1-score is {f1:.3f}, best threshold for binary classification is {threshold:.3f}')
        elif args.binary_threshold is not None:
            threshold = args.binary_threshold
        elif os.path.exists('./temp/threshold'):
            threshold = np.loadtxt('./temp/threshold')
        else:
            threshold = 0.5
        predict = np.where(prob > threshold, 1, 0).tolist()
        predict = [label_map[x] for x in predict]
        output = pd.DataFrame(zip(path_all, prob.tolist(), predict), columns=['file_name', 'probability', f'predict<{label_map[0]} is 0~{threshold:.3f}>'])
    else:
        predict = np.argmax(y_all.cpu().numpy(), axis=1).tolist()
        predict = [label_map[x] for x in predict]
        if args.label_file:
            label_raw = pd.read_csv(args.label_file, header=None, index_col=0).squeeze().to_dict()
            label = [label_raw[x] for x in path_all]
            acc = (predict == label).sum() / len(label)
            logger.info(f'Acc is {acc:.3f}')
        output = pd.DataFrame(zip(path_all, predict), columns=['file_name', 'predict'])
    save_path = os.path.join(args.test_path, 'predict.csv')
    logger.info('Predicts saving......')
    output.to_csv(save_path, index=False)
    logger.info(f'{save_path} saved !!!')
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/swin384.yaml', help='load the config file')
    parser.add_argument('--label_file', type=str, default='', help='for example ./datasets/mimic/test/label.csv, default means only predict and do not calculate accuracy')
    parser.add_argument('--test_path', type=str, default='./datasets/mimic/test', help='test set path')
    parser.add_argument('--ckpt_file', type=str, default='./checkpoints/ckpt_best.pth', help='checkpoint file path')
    parser.add_argument('--mean_std', type=float, nargs='+', default=None, help='input mean and standard deviation if available, for example: 0.483 0.298 or 0.506 0.290')
    parser.add_argument('--cpu_count', type=int, default=None, help='CPU counts, default None means all CPUs')
    parser.add_argument('--batch_size', type=int, default=25, help='setting batch size')
    parser.add_argument('--binary_threshold', type=float, default=None, help='setting threshold for binary classification')
    args = parser.parse_args()
    main(args)