from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

def train_cls_module(config, rank, epoch, net, split_all, loss_fn, train_loader, optimizer, lr_scheduler, scaler, logger, writer):
    net.train()
    if rank == 0:
        iterator = tqdm(train_loader, desc=f'Training epoch {epoch}', dynamic_ncols=True)
    else:
        iterator = train_loader
    num_steps = len(train_loader)
    label_all = torch.tensor([], dtype=torch.int8).cuda(rank)
    idx_all = torch.tensor([], dtype=torch.long)
    y_all = torch.tensor([], dtype=torch.half).cuda(rank)
    conf_all = torch.tensor([])
    h_all = torch.tensor([], dtype=torch.half)
    erm_all = 0
    penalty_all = 0
    for n_iter, (img, label, conf, idx) in enumerate(iterator):
        img = img.cuda(rank, non_blocking=True)
        label = label.cuda(rank, non_blocking=True)
        conf = conf.cuda(rank, non_blocking=True)
        idx = idx.cuda(rank, non_blocking=True)
        with autocast():
            y, h, mix = net(img)
            loss, erm, penalty = loss_fn(y, mix, label, split_all, idx)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dist.all_reduce(erm, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty, op=dist.ReduceOp.SUM)
        erm_reduce = erm.item() / config['world_size']
        penalty_reduce = penalty.item() / config['world_size']
        erm_all += erm.item() * label.shape[0]
        penalty_all += penalty.item() * label.shape[0]
        if rank == 0:
            writer.add_scalar('plot/Learning rate 1', lr, epoch * num_steps + n_iter)
            writer.add_scalar('plot/ERM', erm_reduce, epoch * num_steps + n_iter)
            writer.add_scalar('plot/Penalty', penalty_reduce, epoch * num_steps + n_iter)
        conf_out = torch.zeros(config['world_size'] * conf.shape[0], conf.shape[1]).cuda(rank)
        dist.all_gather_into_tensor(conf_out, conf)
        conf_all = torch.cat((conf_all, conf_out.cpu()), 0)
        h_out = torch.zeros(config['world_size'] * h.shape[0], h.shape[1], h.shape[2], dtype=torch.half).cuda(rank)
        dist.all_gather_into_tensor(h_out, h.half())
        h_all = torch.cat((h_all, h_out.cpu()), 0)
        y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1], dtype=torch.half).cuda(rank)
        dist.all_gather_into_tensor(y_out, y)
        y_all = torch.cat((y_all, y_out), 0)
        idx_out = torch.zeros(config['world_size'] * idx.shape[0], dtype=torch.long).cuda(rank)
        dist.all_gather_into_tensor(idx_out, idx)
        idx_all = torch.cat((idx_all, idx_out.cpu()), 0)
        label_out = torch.zeros(config['world_size'] * label.shape[0], dtype=torch.int8).cuda(rank)
        dist.all_gather_into_tensor(label_out, label)
        label_all = torch.cat((label_all, label_out), 0)
        if (n_iter + 1) % 1 == 0:
            if loss_fn.num_classes == 1:
                prob = torch.sigmoid(y_all).cpu().numpy()
                pr = average_precision_score(label_all.cpu().numpy(), prob)
                auc = roc_auc_score(label_all.cpu().numpy(), prob)
                if rank == 0:
                    writer.add_scalar('plot/PR', pr, epoch * num_steps + n_iter)
                    writer.add_scalar('plot/AUC', auc, epoch * num_steps + n_iter)
            else:
                pred = np.argmax(y_all.cpu().numpy(), axis=1)
                acc = (pred == label_all.cpu().numpy()).sum() / label_all.shape[0]
                if rank == 0:
                    writer.add_scalar('plot/ACC', acc, epoch * num_steps + n_iter)
    erm_mean = erm_all / len(train_loader.dataset)
    penalty_mean = penalty_all / len(train_loader.dataset)
    if loss_fn.num_classes == 1:
        prob = torch.sigmoid(y_all).cpu().numpy()
        pr = average_precision_score(label_all.cpu().numpy(), prob)
        auc = roc_auc_score(label_all.cpu().numpy(), prob)
        logger.info(f'PR for training of epoch {epoch} is {pr:.3f}')
        logger.info(f'AUC for training of epoch {epoch} is {auc:.3f}')
    else:
        pred = np.argmax(y_all.cpu().numpy(), axis=1)
        acc = (pred == label_all.cpu().numpy()).sum() / label_all.shape[0]
        logger.info(f'Acc for training of epoch {epoch} is {acc:.3f}')
    logger.info(f'ERM of epoch {epoch} is {erm_mean:.3f}')
    logger.info(f'Penalty of epoch {epoch} is {penalty_mean:.3f}')
    return conf_all, h_all, idx_all

def train_spl_module(config, rank, epoch, net, loss_fn, env_loader, optimizer, lr_scheduler, scaler, logger, writer):
    net.train()
    if rank == 0:
        iterator = tqdm(env_loader, desc=f'Splitting epoch {epoch}', dynamic_ncols=True)
    else:
        iterator = env_loader
    num_steps = len(env_loader)
    idx_all = torch.tensor([], dtype=torch.long)
    y_all = torch.tensor([], dtype=torch.half)
    loss_all = 0
    for n_iter, (conf, h, idx) in enumerate(iterator):
        conf = conf.cuda(rank, non_blocking=True)
        h = h.cuda(rank, non_blocking=True)
        idx = idx.cuda(rank, non_blocking=True)
        label = torch.tensor(range(idx.shape[0])).cuda(rank)
        with autocast():
            logits_per_table, logits_per_image, y = net(conf, h)
            loss = loss_fn(logits_per_table, logits_per_image, label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_reduce = loss.item() / config['world_size']
        loss_all += loss.item() * idx.shape[0]
        if (n_iter + 1) % 1 == 0:
            if rank == 0:
                writer.add_scalar('plot/Learning rate 2', lr, epoch * num_steps + n_iter)
                writer.add_scalar('plot/NCE', loss_reduce, epoch * num_steps + n_iter)
        y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1], dtype=torch.half).cuda(rank)
        dist.all_gather_into_tensor(y_out, y.half())
        y_all = torch.cat((y_all, y_out.cpu()), 0)
        idx_out = torch.zeros(config['world_size'] * idx.shape[0], dtype=torch.long).cuda(rank)
        dist.all_gather_into_tensor(idx_out, idx)
        idx_all = torch.cat((idx_all, idx_out.cpu()), 0)
    loss_mean = loss_all / len(env_loader.dataset)
    logger.info(f'NCE of epoch {epoch} is {loss_mean:.3f}')   
    split = torch.zeros(torch.max(idx_all)+1, config['n_env'], dtype=torch.long).cuda(rank)
    if rank == 0:
        with threadpool_limits(limits=1):
            pca = PCA(n_components=128)
            y_pca = pca.fit_transform(y_all.numpy())
        gmm = GaussianMixture(n_components=config['n_env'])
        split_all = gmm.fit_predict(y_pca)
        split_all = F.one_hot(torch.from_numpy(split_all)).cuda()
        split[idx_all] = split_all
    dist.barrier()
    dist.broadcast(split, 0)
    return split
    
def validate_module(config, rank, epoch, net, val_loader, logger, writer):
    logger.info(f'Epoch {epoch} begin validating......')
    net.eval()
    if rank == 0:
        iterator = tqdm(val_loader, desc=f'Validating epoch {epoch}', dynamic_ncols=True)
    else:
        iterator = val_loader
    num_steps = len(val_loader)
    with torch.no_grad():
        label_all = torch.tensor([], dtype=torch.int8).cuda(rank)
        y_all = torch.tensor([], dtype=torch.half).cuda(rank)
        for n_iter, (img, label) in enumerate(iterator):
            img = img.cuda(rank, non_blocking=True)
            label = label.cuda(rank, non_blocking=True)
            with autocast():
                y, _, _ = net(img)
            y_out = torch.zeros(config['world_size'] * y.shape[0], y.shape[1], dtype=torch.half).cuda(rank)
            dist.all_gather_into_tensor(y_out, y)
            y_all = torch.cat((y_all, y_out), 0)
            label_out = torch.zeros(config['world_size'] * label.shape[0], dtype=torch.int8).cuda(rank)
            dist.all_gather_into_tensor(label_out, label)
            label_all = torch.cat((label_all, label_out), 0)
            if (n_iter + 1) % 1 == 0:
                if net.module.num_classes == 1:
                    prob = torch.sigmoid(y_all).detach().cpu().numpy()
                    pr = average_precision_score(label_all.cpu().numpy(), prob)
                    auc = roc_auc_score(label_all.cpu().numpy(), prob)
                    if rank == 0:
                        writer.add_scalar('plot/val-PR', pr, epoch * num_steps + n_iter)
                        writer.add_scalar('plot/val-AUC', auc, epoch * num_steps + n_iter)
                else:
                    pred = np.argmax(y_all.detach().cpu().numpy(), axis=1)
                    acc = (pred == label_all.detach().cpu().numpy()).sum() / label_all.shape[0]
                    if rank == 0:
                        writer.add_scalar('plot/val-ACC', acc, epoch * num_steps + n_iter)
        if net.module.num_classes == 1:
            prob = torch.sigmoid(y_all).detach().cpu().numpy()
            pr = average_precision_score(label_all.cpu().numpy(), prob)
            auc = roc_auc_score(label_all.detach().cpu().numpy(), prob)
            logger.info(f'PR for validation of epoch {epoch} is {pr:.3f}')
            logger.info(f'AUC for validation of epoch {epoch} is {auc:.3f}')
            acc_or_auc = auc
        else:
            pred = np.argmax(y_all.detach().cpu().numpy(), axis=1)
            acc = (pred == label_all.cpu().numpy()).sum() / label_all.shape[0]
            logger.info(f'Acc for validation of epoch {epoch} is {acc:.3f}')
            acc_or_auc = acc
        logger.info(f'Epoch {epoch} validating finished !!!')
    return acc_or_auc