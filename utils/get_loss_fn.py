import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class irm_loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    @autocast()
    def forward(self, outputs, mix, labels, split_all, idx):
        scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
        if self.num_classes == 1:
            loss_value = F.binary_cross_entropy_with_logits(outputs.squeeze(1), labels.float(), reduction='none')
            penalty = F.binary_cross_entropy_with_logits((outputs * scale).squeeze(1), labels.float(), reduction='none')
            mix_loss = F.binary_cross_entropy_with_logits(mix.squeeze(1), labels.float())
        else:
            loss_value = F.cross_entropy(outputs, labels, reduction='none')
            penalty = F.cross_entropy(outputs * scale, labels, reduction='none')
            mix_loss = F.cross_entropy(mix, labels)
        split = split_all[idx]
        penalty = (split * penalty.unsqueeze(-1) / penalty.shape[0]).sum(0)
        erm_risk = (split * loss_value.unsqueeze(-1) / loss_value.shape[0]).sum(0)
        irm_risk_list = []
        for index in range(penalty.size(0)):
            irm_risk = torch.autograd.grad(penalty[index], [scale], create_graph=True)[0]
            irm_risk_list.append(torch.sum(irm_risk ** 2))
        erm = erm_risk.sum()
        penalty = torch.stack(irm_risk_list).sum()
        irm_risk_final = erm + 1e2 * penalty + mix_loss
        return irm_risk_final, erm, penalty

class nce_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @autocast()
    def forward(self, logits_per_table, logits_per_image, label):
        loss_table = F.cross_entropy(logits_per_table, label)
        loss_image = F.cross_entropy(logits_per_image, label)
        loss = (loss_image + loss_table) / 2
        return loss