import torch
import torch.nn as nn
from torch.nn import init

from . import blocks


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #if not amp:# modified
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class MLP_Head(nn.Module):
    def __init__(self, use_mlp=True, init_type='xavier', init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(MLP_Head, self).__init__()
        self.l2norm = blocks.Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        
    def create_mlp(self, feats, device):
        for mlp_id, feat in enumerate(feats):
            #JB_EDIT
            try:
                if len(feat.shape) == 4: #Conv 
                    input_nc = feat.shape[1]
                else: #Attention 
                    input_nc = feat.shape[2]
            except:
                feat = feat[0]
                if len(feat.shape) == 4: #Conv 
                    input_nc = feat.shape[1]
                else: #Attention 
                    input_nc = feat.shape[2]
                
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])

            
            mlp.to(device)
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, [0])#0 -> device
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, contrastive_mlp=True):
        return_ids = []
        return_feats = []

        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):
            #JB_EDIT
            #if not self.if_inst_feat:
            if len(feat.shape) == 4: #Conv 
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3] #torch.Size([2, 128, 256, 256])    #inst 2,256,8,8
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) #torch.Size([2, 65536, 128]) ##inst [2, 64, 256]
            else: #Attention 
                feat_reshape = feat ##여기수정

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape #inst 128,256
                patch_id = []

            if self.use_mlp and contrastive_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                
                x_sample = x_sample.cuda().to(feats[0].device)
                x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)


        return return_feats, return_ids