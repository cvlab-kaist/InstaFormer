import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks
     
    # def forward(self, x):
    #     for attn, ff in self.layers:
    #         x = attn(x) + x
    #         x = ff(x) + x
    #     return x
    
    def forward(self, x, mask=None):
        i = 0
        ll = []
        for attn, ff in self.layers:
            if mask == None:
                x = attn(x) + x
            else:
                x = attn.fn(attn.norm(x),mask) + x
                
            x = ff(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1
        self.features = tuple(ll)
    
        return x
   
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, hybrid=False, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size #pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.__image_size = image_size
        self.__patch_size = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        if hybrid:
            backbone = ResNetV2(
                layers=(3,4,9), num_classes=0, global_pool='', in_chans=channels,
                preact=False, stem_type='same')
            self.to_patch_embedding = HybridEmbed(
                    backbone, img_size=image_size, in_chans=channels, embed_dim=dim)
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim,eps=1e-06),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_image_size(self):
        return self.__image_size
    def get_patch_size(self):
        return self.__patch_size

    def resize_pos_embed(self, h, w, start_index=1):
        old_h, old_w = self.__image_size
        self.__image_size = (h,w)
        pw, ph = self.get_patch_size()
        gs_w = w//pw
        gs_h = h//ph
        
        posemb_tok, posemb_grid = (
            self.pos_embedding[:, : start_index],
            self.pos_embedding[0, start_index :],
        )

                
        gs_old_w = old_w // pw
        gs_old_h = old_h // ph

        posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        self.pos_embedding = nn.Parameter(torch.cat([posemb_tok, posemb_grid], dim=1))