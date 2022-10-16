import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align # Instance Aware
from einops.layers.torch import Rearrange

from . import vit
from . import blocks

class PreInstanceNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = blocks.AdaptiveInstanceNorm2d(dim,eps=1e-06)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class AdaIn_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreInstanceNorm(dim, vit.Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreInstanceNorm(dim, vit.FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks

    def forward(self, x):
        i = 0
        ll = []
        for attn, ff in self.layers:
            x = attn(x) + x                
            x = ff(x) + x

            if i in self.hooks:
                ll.append(x)
            i += 1

        self.features = tuple(ll)
    
        return x

class Transformer_Aggregator(nn.Module):
    def __init__(self, img_size=88, patch_size=8, embed_C=1024, feat_C=256, depth=6, heads=4, mlp_dim=4096):
        super(Transformer_Aggregator, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = blocks.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=feat_C, embed_dim= embed_C)
        self.box_embed = blocks.PatchEmbed(
            img_size=patch_size, patch_size=patch_size, in_chans=feat_C, embed_dim= embed_C)

        self.pos_embed_x = blocks.PositionalEncoding(d_model=embed_C//4, dropout=0., max_len=img_size)
        self.pos_embed_y = blocks.PositionalEncoding(d_model=embed_C//4, dropout=0., max_len=img_size)
        self.pos_embed_h = blocks.PositionalEncoding(d_model=embed_C//4, dropout=0., max_len=img_size)
        self.pos_embed_w = blocks.PositionalEncoding(d_model=embed_C//4, dropout=0., max_len=img_size)        
        
        pos_embed = torch.cat((
                self.pos_embed_x()[..., None, :].repeat(1, 1, img_size, 1), # 
                self.pos_embed_y()[:, None].repeat(1, img_size, 1, 1), #
                self.pos_embed_w()[:, (8 - 1)][:, None, None].repeat(1, img_size, img_size, 1),    
                self.pos_embed_h()[:, (8 - 1)][:, None, None].repeat(1, img_size, img_size, 1),
                ), dim=3)
        self.pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2), size=(img_size//patch_size, img_size//patch_size), mode='bilinear').flatten(2).transpose(-1, -2)

        self.transformer = AdaIn_Transformer(embed_C, depth, heads, feat_C, mlp_dim, dropout=0.)

    def extract_box_feature(self, out, box, num_box):
        b, c, h, w = out.shape
        batch_index = torch.arange(0.0, b).repeat(num_box).view(num_box, -1).transpose(0,1).flatten(0,1).to(out.device)
        roi_box_info = box.view(-1,5).to(out.device) 

        roi_info = torch.stack((batch_index, \
                            roi_box_info[:, 1] * w, \
                            roi_box_info[:, 2] * h, \
                            roi_box_info[:, 3] * w, \
                            roi_box_info[:, 4] * h), dim = 1).to(out.device)
        aligned_out = roi_align(out, roi_info, 8) 

        aligned_out.view(b, num_box, c, 8, 8)[torch.where(box[:,:,0] == -1)] = 0
        aligned_out.view(-1, c, 8, 8)        

        return aligned_out
    
    def add_box(self, out, box, box_info, num_box, pos_embed=None): 
        b = out.shape[0]
        box = self.box_embed(box).squeeze().view(b, num_box, -1) 
        
        x_coord = (box_info[..., 1::2].mean(dim=2) * (self.img_size - 1)).long() 
        y_coord = (box_info[..., 2::2].mean(dim=2) * (self.img_size - 1)).long() 
        w = ((box_info[..., 3] - box_info[..., 1]) * (self.img_size - 1)).long()
        h = ((box_info[..., 4] - box_info[..., 2]) * (self.img_size - 1)).long()
        
        box_pos_embed = torch.cat((
            self.pos_embed_x()[..., None, :].repeat(1, 1, self.img_size, 1), 
            self.pos_embed_y()[:, None].repeat(1, self.img_size, 1, 1), 
            ), dim=3).squeeze()

        box += torch.cat((
                box_pos_embed[y_coord, x_coord], box_pos_embed[h, w] 
                ), dim=2)

        added_out = torch.cat((out, box), dim=1) 
        
        return added_out

    def forward(self, x, box_info=None, num_box=-1):
        embed_x = self.patch_embed(x) + self.pos_embed.to(x.device)
        
        if box_info != None:
            box_feat = self.extract_box_feature(x, box_info, num_box)
            embed_x = self.add_box(embed_x, box_feat, box_info, num_box)
        
        out = self.transformer(embed_x)
        
        if num_box > 0:
            aggregated_feat, aggregated_box = out[:,:-num_box,:], out[:, -num_box:, :]
        else:
            aggregated_feat, aggregated_box = out, None 
        
        return aggregated_feat, aggregated_box
        
        
