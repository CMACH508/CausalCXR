from torch.nn import TransformerDecoderLayer, TransformerDecoder, LayerNorm
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
import torch

class decoder(nn.Module):
    def __init__(self, config, dim_conf):
        super().__init__()
        self.img_size = config['img_size']
        self.embedding = nn.Linear(dim_conf, 1536)
        decoder_layer = TransformerDecoderLayer(d_model=1536, 
                                                nhead=8, 
                                                dim_feedforward=2048, 
                                                dropout=0.1,                                                    
                                                activation=F.relu, 
                                                layer_norm_eps=1e-5, 
                                                batch_first=True,
                                                norm_first=False)
        decoder_norm = LayerNorm(1536, 1e-5)
        self.decoder = TransformerDecoder(decoder_layer, 1, decoder_norm)
        self.attn = nn.MultiheadAttention(1536, 1)
        for para in self.attn.out_proj.parameters():
            para.requires_grad = False
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
    
    @autocast()
    def forward(self, x, memory):
        x = self.embedding(x).unsqueeze(1)
        x = self.decoder(x, memory)
        size = x.shape[0]
        weights = torch.zeros(size, size, self.img_size ** 2 // 1024).cuda()
        for i in range(size):
            for j in range(size):
                _, weight = self.attn(x[i], memory[j], memory[j], need_weights=True)
                weights[i, j]=weight.squeeze()
        memory = (memory / memory.norm(dim=2, keepdim=True)).permute(1, 2, 0)
        x = (x / x.norm(dim=2, keepdim=True)).permute(1, 0, 2)
        prod = torch.matmul(x, memory).permute(1, 2, 0) * weights
        logit_scale = self.logit_scale.exp()
        logits_per_table = logit_scale * torch.sum(prod, dim = 2)
        logits_per_image = logits_per_table.t()
        x = x.squeeze()
        return logits_per_table, logits_per_image, x