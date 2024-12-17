from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.get_transform import mean_std_transform
from utils.build_data import train_dataset, test_dataset

def get_train_mean_std(path, n_cpus):
    transform = mean_std_transform()
    dataset = train_dataset(root=path, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_cpus)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data, _, _, _ in tqdm(dataloader, desc='Calculating', dynamic_ncols=True):
        for dim in range(3):
            mean[dim] += data[:, dim, :, :].mean()
            std[dim] += data[:, dim, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return list(mean.numpy()), list(std.numpy())

def get_test_mean_std(path, n_cpus):
    transform = mean_std_transform()
    dataset = test_dataset(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_cpus)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data, _ in tqdm(dataloader, desc='Analysing', dynamic_ncols=True):
        for dim in range(3):
            mean[dim] += data[:, dim, :, :].mean()
            std[dim] += data[:, dim, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return list(mean.numpy()), list(std.numpy())