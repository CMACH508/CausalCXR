from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, SequentialLR

def build_all_lrs(opt):
    lrs_cls = build_lrs(opt[0], milestone=500, T_max=500)
    lrs_spl = build_lrs(opt[1], milestone=100, T_max=100)
    return lrs_cls, lrs_spl

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
def build_lrs(optimizer, milestone, T_max):
    scheduler1 = WarmUpLR(optimizer, milestone)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=T_max)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[milestone])
    return scheduler