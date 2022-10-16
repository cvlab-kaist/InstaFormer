import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks

class ContentEncoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64, n_downsampling = 2, no_antialias=False):
        super(ContentEncoder, self).__init__()
        
        model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(ngf * mult * 2),
                        nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.InstanceNorm2d(ngf * mult * 2),
                        nn.ReLU(True),
                        blocks.Downsample(ngf * mult * 2)]

        self.model = nn.Sequential(*model)

    def forward(self, x, layers=[]):
        features=[]
        feature = x

        for i, layer in enumerate(self.model):
            feature = layer(feature)
            
            if i in layers:
                features.append(feature)
        out = feature 
        return out, features
        
# ContentEncoder(
# (model): Sequential(
#     (0): ReflectionPad2d((3, 3, 3, 3))
#     (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
#     (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (3): ReLU(inplace=True)
#     (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (6): ReLU(inplace=True)
#     (7): Downsample(
#     (pad): ReflectionPad2d([1, 1, 1, 1])
#     )
#     (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (10): ReLU(inplace=True)
#     (11): Downsample(
#     (pad): ReflectionPad2d([1, 1, 1, 1])
#     )
# )
# )


#MGUIT
# class ContentEncoder(nn.Module):
#     def __init__(self, input_dim_a=3, tch=64):
#         super(ContentEncoder, self).__init__()
#         encA_c = []

#         encA_c += [blocks.LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
#         for i in range(1, 3):
#             encA_c += [blocks.ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
#             tch *= 2
        
#         self.convA = nn.Sequential(*encA_c)
        
#     def forward(self, xa): #xa: 8,3,256,256
#         outputA = self.convA(xa)
#         return F.normalize(outputA, dim=1)

# (convA): Sequential(
#       (0): LeakyReLUConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((3, 3, 3, 3))
#           (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
#           (2): LeakyReLU(negative_slope=0.01, inplace=True)
#         )
#       )
#       (1): ReLUINSConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((1, 1, 1, 1))
#           (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
#           (2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#           (3): ReLU(inplace=True)
#         )
#       )
#       (2): ReLUINSConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((1, 1, 1, 1))
#           (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
#           (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#           (3): ReLU(inplace=True)
#         )
#       )
#     )