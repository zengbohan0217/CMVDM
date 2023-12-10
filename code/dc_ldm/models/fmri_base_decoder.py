import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dc_ldm.modules.diffusionmodules.util import zero_module


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def norm_depth_01(depth):
    # Normalize to 0-1.
    # depth is NxHxW
    depth_min = depth.view(*depth.shape[:-2], -1).min(-1).values[:, None, None]
    depth_max = depth.view(*depth.shape[:-2], -1).max(-1).values[:, None, None]
    depth = (depth - depth_min) / (depth_max - depth_min)
    return depth


class BaseDecoder(nn.Module):
    def __init__(self, in_dim, out_img_res, start_CHW=(64, 14, 14), n_conv_layers_ramp=3, n_chan=64, n_chan_output=3, depth_extractor=None):
        super(BaseDecoder, self).__init__()

        self.start_CHW = start_CHW
        upsample_scale_factor = (out_img_res / start_CHW[-1]) ** (1/n_conv_layers_ramp)
        self.input_fc = nn.Linear(in_dim, np.prod(self.start_CHW))

        kernel_size = 5

        pad_size = int(kernel_size // 2)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'),
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(start_CHW[0], n_chan, kernel_size),
            nn.GroupNorm(32, n_chan),
            MemoryEfficientSwish(),
        ) for block_index in range(n_conv_layers_ramp)] + \
        [nn.Sequential(
            nn.Conv2d(start_CHW[0], n_chan, kernel_size, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_chan)
        ) for _ in range(0)])

        self.top = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(n_chan, n_chan_output, kernel_size),
            nn.Sigmoid()
        )

        self.depth_extractor = depth_extractor
        self.trainable = [self.input_fc, self.blocks, self.top]
        self.fmri_dim=in_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x[:, :self.fmri_dim]
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        for block_index, block in enumerate(self.blocks):
            x = block(x)

        x = self.top(x)

        if self.depth_extractor:
            x_depth = self.depth_extractor(x)
            x_depth = norm_depth_01(x_depth).unsqueeze(1)
            x = torch.cat([x, x_depth], 1)
        x = F.interpolate(x, size=(512, 512))

        return x


class text_clip_encoder(nn.Module):
    def __init__(self, cond_dim=512, clip_dim=512):
        super(text_clip_encoder, self).__init__()
        inner_mlp_dim = 1024
        self.extend_deal = nn.Sequential(nn.Linear(cond_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, cond_dim),
                                 nn.SiLU())
        self.zero_linear = zero_module(nn.Linear(cond_dim, cond_dim))

        # define clip matcher
        self.clip_pred_conv = nn.Sequential(
                                 nn.Conv1d(77, 64, 3, padding=1, bias=True),
                                 nn.Conv1d(64, 4, 3, padding=1, bias=True))
        self.clip_matcher_ = nn.Sequential(nn.Linear(cond_dim*4, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, clip_dim),
                                 nn.SiLU())
    
    def forward(self, encode_c):
        return self.zero_linear(self.extend_deal(encode_c))
    
    def get_clip(self, encode_c):
        out = self.extend_deal(encode_c)
        out = self.clip_pred_conv(out).view(out.size(0), -1)
        return self.clip_matcher_(out)
