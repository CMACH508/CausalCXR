import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils.get_transform import train_transform, test_transform

class train_dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data_dir = root
        self.train = train
        self.transform = transform        
        tr_lab_dir = os.path.join(self.data_dir, 'train/label.txt')
        val_lab_dir = os.path.join(self.data_dir, 'val/label.txt')
        label_train = np.loadtxt(tr_lab_dir, dtype=str)
        label_val = np.loadtxt(val_lab_dir, dtype=str)
        label_train = pd.Categorical(label_train)
        self.label_map = label_train.categories
        self.label_train = label_train.codes
        self.label_val = pd.Categorical(label_val, categories=self.label_map).codes
        tr_conf_dir = os.path.join(self.data_dir, 'train/confounder.csv')
        self.conf_train = pd.read_csv(tr_conf_dir)
        numeric_features = self.conf_train.dtypes[self.conf_train.dtypes != 'object'].index
        self.conf_train[numeric_features] = stats.zscore(self.conf_train[numeric_features], nan_policy='omit')
        self.conf_train[numeric_features] = self.conf_train[numeric_features].fillna(0)
        self.conf_train = pd.get_dummies(self.conf_train, drop_first=True).to_numpy()

    def __getitem__(self, index):
        if self.train:
            path_img = os.path.join(self.data_dir, f'train/{index}.jpg')
            label = self.label_train[index]
            conf = self.conf_train[index].astype('single')
            img = Image.open(path_img)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, label, conf, index
        else:
            path_img = os.path.join(self.data_dir, f'val/{index}.jpg')
            label = self.label_val[index]
            img = Image.open(path_img)
            img = img.convert('RGB')
            if self.transform is not None: 
                img = self.transform(img)      
            return img, label

    def __len__(self):
        if self.train:                
            length = len(self.label_train)
        else:
            length = len(self.label_val)
        return length

class test_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_dir = root
        self.transform = transform
        self.walk_path = glob.glob(os.path.join(self.data_dir, '*.jpg'))
        
    def __getitem__(self, index):
        path_img = self.walk_path[index]
        img = Image.open(path_img)
        img = img.convert('RGB')
        if self.transform is not None: 
            img = self.transform(img)    
        return img, os.path.basename(path_img)
    
    def __len__(self):
        length = len(self.walk_path)
        return length
    
class env_dataset(Dataset):
    def __init__(self, conf, h, idx):
        self.conf = conf
        self.h = h
        self.idx = idx
        
    def __getitem__(self, index):
        loc = torch.nonzero(self.idx==index)[0].squeeze()
        conf = self.conf[loc]
        h = self.h[loc]
        return conf, h, index
    
    def __len__(self):
        length = torch.max(self.idx) + 1
        return length.int()
    
class build_trainloader:
    def __init__(self, rank, config):
        self.batch_size = config['batch_size']
        self.path = os.path.join(config['data_path'], config['train_set'])
        if config['mean'] is not None:
            self.mean = config['mean']
        elif os.path.exists('./temp/mean'):
            self.mean = np.loadtxt('./temp/mean')[0]
        else:
            self.mean = 0.5
        if config['std'] is not None:
            self.std = config['std']
        elif os.path.exists('./temp/std'):
            self.std = np.loadtxt('./temp/std')[0]
        else:
            self.std = 0.5
        self.size = config['img_size']
        self.num_workers = config['num_workers']
        tr_transform = train_transform(self.mean, self.std, self.size)         
        self.train_set = train_dataset(root=self.path, train=True, transform=tr_transform)
        label_map = self.train_set.label_map
        if rank == 0:
            pd.DataFrame(label_map).to_csv('./temp/label_map', header=False, index=False)
        self.num_classes = label_map.shape[0]
        if self.num_classes == 2:
            self.num_classes = 1
        self.dim_conf = self.train_set.conf_train.shape[1]
        self.num_train = len(self.train_set)
        
    def train_loader(self):
        train_sampler = DistributedSampler(self.train_set)
        train_loader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=train_sampler
        )
        return train_loader
    
    def val_loader(self):
        val_transform = test_transform(self.mean, self.std, self.size)
        val_set = train_dataset(root=self.path, train=False, transform=val_transform)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(
            val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=val_sampler
        )
        return val_loader

class build_testloader:
    def __init__(self, config, mean, std, batch_size):
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.size = config['img_size']
        label_map = np.loadtxt('./temp/label_map', dtype=str)
        self.num_classes = label_map.shape[0]
        if self.num_classes == 2:
            self.num_classes = 1
        
    def test_loader(self, path, num_workers):
        te_transform = test_transform(self.mean, self.std, self.size)
        test_set = test_dataset(root=path, transform=te_transform)
        test_loader = DataLoader(
            test_set, 
            batch_size=self.batch_size, 
            num_workers=num_workers
        )
        return test_loader
    
def build_envloader(config, conf, h, idx):
    env_set = env_dataset(conf, h, idx)
    env_sampler = DistributedSampler(env_set)
    env_loader = DataLoader(
        env_set, 
        batch_size=config['batch_size_spl'], 
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        sampler=env_sampler
    )
    return env_loader