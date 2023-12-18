
import src
from src.utils.datasets import RGBD_Dataset
from src.utils.misc import (cprint1, cprintc, cprintm, np,
                                           interpolate, extract_patches,
                                           norm_depth_01, tup2list, hw_flatten)
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import pretrainedmodels as pm
import torchvision.models as torchvis_models
from pretrainedmodels import utils as pmutils
from src.config import *
from src.midas.midas_net import MidasNet
from src.midas.midas_net_custom import MidasNet_small
from absl import flags
identity = lambda x: x

FLAGS = flags.FLAGS

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)



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

class MultiBranch(nn.Module):
    def __init__(self, model, branch_dict, main_branch, spatial_out_dims=20, replace_maxpool=False):
        super(MultiBranch, self).__init__()
        name_to_module = dict(model.named_modules())
        self.branch_dict = branch_dict
        self.target_modules = list(branch_dict.values())
        self.main_branch = main_branch
        self.adapt_avg_pool_suffix = '_adapt_avg_pool'
        if spatial_out_dims is not None and isinstance(spatial_out_dims, int):
            spatial_out_dims = dict(zip(self.target_modules, [spatial_out_dims] * len(self.target_modules)))

        for module_name in main_branch:
            module = name_to_module[module_name]
            if replace_maxpool and isinstance(module, nn.MaxPool2d):
                module = nn.Upsample(scale_factor=.5)
            self.add_module(module_name.replace('.', '_'), module)
        for module_name in self.target_modules:
            if spatial_out_dims is not None:
                module = nn.AdaptiveAvgPool2d(spatial_out_dims[module_name])
                self.add_module(module_name + self.adapt_avg_pool_suffix, module)

    def __getitem__(self, module_name):
        return getattr(self, module_name.replace('.', '_'), None)

    def num_output_planes(self):
        n_planes = []
        for target_module in self.target_modules:
            for module_name in self.main_branch[:self.main_branch.index(target_module)+1][::-1]:
                try:
                    n_planes.append(list(self[module_name].parameters())[0].shape[0])
                    break
                except:
                    pass
        return n_planes

    def forward(self, x):
        X = {}
        for module_name in self.main_branch:
            if isinstance(self[module_name], nn.Linear) and x.ndim > 2:
                x = x.view(len(x), -1)
            x = self[module_name](x)
            if module_name in self.target_modules: # Collect
                X[module_name] = x.clone()
                avg_pool = self[module_name + self.adapt_avg_pool_suffix]
                if avg_pool:
                    X[module_name] = avg_pool(X[module_name])
        return list(X.values())

class BaseEncoderVGG19ml(nn.Module):
    def __init__(self, out_dim, random_crop_pad_percent, spatial_out_dim=None, drop_rate=0.25):
        super(BaseEncoderVGG19ml, self).__init__()
        self.drop_rate = drop_rate
        cprintm('(*) Backbone: {}'.format('vgg19'))
        bbn = pm.__dict__['vgg19'](num_classes=1000, pretrained='imagenet')
        self.img_xfm_basic = pmutils.TransformImage(bbn, scale=1)
        self.img_xfm_train = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BILINEAR),
            transforms.RandomCrop(size=224, padding=int(random_crop_pad_percent / 100 * 224), padding_mode='edge'),
            *self.img_xfm_basic.tf.transforms[-4:],
        ])

        branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
            'conv1': ['_features.{}'.format(i) for i in range(4)],
            'conv2': ['_features.{}'.format(i) for i in range(9)],
            'conv3': ['_features.{}'.format(i) for i in range(18)],
            'conv4': ['_features.{}'.format(i) for i in range(27)],
        }

        spatial_out_dims = None

        main_branch = list(branch_dict.values())[-1]
        branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
        self.multi_branch_bbn = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)

        self.bbn_n_out_planes = self.multi_branch_bbn.num_output_planes()
        self.out_shapes = [(48, 14, 14)]
        self.n_out_planes = self.out_shapes[0][0]
        in_dim = np.prod(self.out_shapes[0])

        kernel_size = 3
        pad_size = int(kernel_size // 2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[0]),
            nn.MaxPool2d(2),

            nn.Conv2d(self.bbn_n_out_planes[0], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),

            nn.Conv2d(self.n_out_planes, self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[1]),

            nn.Conv2d(self.bbn_n_out_planes[1], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),

            nn.Conv2d(self.n_out_planes, self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[2]),

            nn.Conv2d(self.bbn_n_out_planes[2], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[3]),

            nn.Conv2d(self.bbn_n_out_planes[3], self.n_out_planes, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )
        self.bn_sum = nn.BatchNorm2d(self.n_out_planes)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_head = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.trainable = self.convs + [self.bn_sum, self.fc_head]


    def forward_bbn(self, x, detach_bbn=False):
        x = interpolate(x, size=im_res(), mode=FLAGS.interp_mode)
        X = self.multi_branch_bbn(x)
        if detach_bbn:
            X = [xx.detach() for xx in X]
        feats_dict = dict(zip(self.multi_branch_bbn.branch_dict.keys(), X))
        return feats_dict

    def forward_convs(self, feats_dict):
        X = [conv(xx) for xx, conv in zip(feats_dict.values(), self.convs)]
        return X

    def forward(self, x, feats=False, detach_bbn=False):
        feats_dict = self.forward_bbn(x, detach_bbn=detach_bbn)
        X = self.forward_convs(feats_dict)
        x = torch.stack(X).sum(0)
        x = self.bn_sum(x)
        x = self.dropout(x)
        x = self.fc_head(x.view(x.size(0), -1))
        if feats:
            return x, feats_dict
        else:
            return x

class SeparableEncoderVGG19ml(nn.Module):
    def __init__(self, out_dim, random_crop_pad_percent, spatial_out_dim=None, drop_rate=0.25):
        super(SeparableEncoderVGG19ml, self).__init__()
        self.drop_rate = drop_rate
        cprintm('(*) Backbone: {}'.format('vgg19'))

        if FLAGS.is_rgbd:
            bbn = torchvis_models.__dict__['vgg19'](pretrained=True)
        else:
            bbn = pm.__dict__['vgg19'](num_classes=1000, pretrained='imagenet')

        if FLAGS.is_rgbd:
            if FLAGS.is_rgbd == 1:  # RGBD
                bbn.features = nn.Sequential(
                    nn.Conv2d(4, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_rgbd_large_norm_within_img'
            else:  # Depth only
                bbn.features = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_depth_only_large_norm_within_img'

            cprint1('   >> Loading Encoder bbn checkpoint: {}'.format(ckpt_name))
            state_dict_loaded = torch.load(f'{PROJECT_ROOT}/data/imagenet_rgbd/{ckpt_name}_best.pth.tar')['state_dict']
            state_dict_loaded = { k.replace('module.', ''): v for k, v in state_dict_loaded.items() }
            bbn.load_state_dict(state_dict_loaded)

            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['features.{}'.format(i) for i in range(4)],
                'conv2': ['features.{}'.format(i) for i in range(9)],
                'conv3': ['features.{}'.format(i) for i in range(18)],
                'conv4': ['features.{}'.format(i) for i in range(27)],
            }
        else:
            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['_features.{}'.format(i) for i in range(4)],
                'conv2': ['_features.{}'.format(i) for i in range(9)],
                'conv3': ['_features.{}'.format(i) for i in range(18)],
                'conv4': ['_features.{}'.format(i) for i in range(27)],
            }

        spatial_out_dims = None
        main_branch = list(branch_dict.values())[-1]
        branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
        self.multi_branch_bbn = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)

        self.bbn_n_out_planes = self.multi_branch_bbn.num_output_planes()
        self.patch_size = 3
        self.out_shapes = [(32, 28 - self.patch_size + 1, 28 - self.patch_size + 1)] * 3 + [(32, 14 - self.patch_size + 1, 14 - self.patch_size + 1)]
        self.n_out_planes = self.out_shapes[0][0]
        kernel_size = 3
        pad_size = int(kernel_size // 2)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[0]),

            nn.MaxPool2d(2),

            nn.Conv2d(self.bbn_n_out_planes[0], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[1]),

            nn.Conv2d(self.bbn_n_out_planes[1], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[2]),

            nn.Conv2d(self.bbn_n_out_planes[2], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[3]),

            nn.Conv2d(self.bbn_n_out_planes[3], self.n_out_planes, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

        # Separable part
        self.space_maps = nn.ModuleDict({
            str(in_space_dim): nn.Linear(in_space_dim**2, out_dim, bias=False) for in_space_dim in np.unique(tup2list(self.out_shapes, 1))
        })

        self.chan_mixes = nn.ModuleList([ChannelMix(self.n_out_planes*self.patch_size**2, out_dim) for _ in range(len(self.convs))])
        self.branch_mix = nn.Parameter(torch.Tensor(out_dim, len(self.chan_mixes)))
        self.branch_mix.data.fill_(1.)

        self.dropout = nn.Dropout(drop_rate)

        self.trainable = self.convs + list(self.space_maps.values()) + list(self.chan_mixes) + [self.branch_mix]

    def forward_bbn(self, x, detach_bbn=False):
        x = interpolate(x, size=im_res(), mode=FLAGS.interp_mode)
        X = self.multi_branch_bbn(x)
        if detach_bbn:
            X = [xx.detach() for xx in X]
        feats_dict = dict(zip(self.multi_branch_bbn.branch_dict.keys(), X))
        return feats_dict

    def forward_convs(self, feats_dict):
        X = [conv(xx) for xx, conv in zip(feats_dict.values(), self.convs)]
        return X

    def forward(self, x, feats=False, detach_bbn=False):
        feats_dict = self.forward_bbn(x, detach_bbn=detach_bbn)
        X = self.forward_convs(feats_dict)
        X = [extract_patches(x, self.patch_size) for x in X]
        X = [self.space_maps[str(x.shape[-1])](hw_flatten(x)) for x in X]  # => BxCxV
        X = [self.dropout(x) for x in X]
        X = [f(x) for f, x in zip(self.chan_mixes, X)]

        x = torch.stack(X, dim=-1)
        x = (x * self.branch_mix.abs()).sum(-1)
        if feats:
            return x, feats_dict
        else:
            return x

class ChannelMix(nn.Module):
    def __init__(self, n_chan, out_dim):
        super(ChannelMix, self).__init__()
        self.chan_mix = nn.Parameter(torch.Tensor(out_dim, n_chan))
        nn.init.xavier_normal(self.chan_mix)

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.bias.data.fill_(0.01)

    def forward(self, x):
        # BxCxN
        x = (x * self.chan_mix.T).sum(-2)
        x += self.bias
        # BxN
        return x

class BaseDecoder(nn.Module):
    def __init__(self, in_dim, out_img_res, start_CHW=(64, 14, 14), n_conv_layers_ramp=3, n_chan=64, n_chan_output=3, depth_extractor=None):
        super(BaseDecoder, self).__init__()

        self.start_CHW = start_CHW
        upsample_scale_factor = (out_img_res / start_CHW[-1]) ** (1/n_conv_layers_ramp)
        self.input_fc = nn.Linear(in_dim, np.prod(self.start_CHW))

        kernel_size = 5

        pad_size = int(kernel_size // 2)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'), #FLAGS.interp_mode),
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

    def forward(self, x):
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        for block_index, block in enumerate(self.blocks):
            x = block(x)

        x = self.top(x)

        if self.depth_extractor:
            x_depth = self.depth_extractor(x)
            x_depth = norm_depth_01(x_depth).unsqueeze(1)
            x = torch.cat([x, x_depth], 1)

        return x

class NewBaseDecoder(nn.Module): 
    def __init__(self, in_dim, out_img_res, start_CHW=(8, 14, 14), n_conv_layers_ramp=3, n_chan=32, n_chan_output=3, depth_extractor=None):
        super(BaseDecoder, self).__init__()

        self.start_CHW = start_CHW
        # upsample_scale_factor = (out_img_res / start_CHW[-1]) ** (1/n_conv_layers_ramp)
        self.input_fc = nn.Linear(in_dim, np.prod(self.start_CHW))

        # kernel_size = 3

        # pad_size = int(kernel_size // 2)
        # self.blocks = nn.ModuleList([nn.Sequential(
        #     nn.Upsample(scale_factor=upsample_scale_factor, mode='nearest'), #FLAGS.interp_mode),
        #     # nn.ReflectionPad2d(pad_size),
        #     nn.Conv2d(start_CHW[0], n_chan, kernel_size+block_index*2, padding=pad_size+block_index),
        #     nn.ReflectionPad2d(pad_size),
        #     nn.GroupNorm(32, n_chan),
        #     MemoryEfficientSwish(),
        # ) for block_index in range(n_conv_layers_ramp)] + \
        # [nn.Sequential(
        #     nn.Conv2d(start_CHW[0], n_chan, kernel_size, padding=pad_size),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(n_chan)
        # ) for _ in range(1)])
        
        # self.resblocks1 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)
        # self.att1 = make_attn(n_chan)
        # self.resblocks2 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)

        self.resblocks3 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)
        self.att2 = make_attn(n_chan)
        self.resblocks4 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)
        
        # self.resblocks5 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)
        # self.att3 = make_attn(n_chan)
        # self.resblocks6 = ResnetBlock(in_channels=n_chan,out_channels=n_chan,temb_channels=0,dropout=0.25)

        

        self.block1 = nn.Sequential(
            nn.Conv2d(self.start_CHW[0], n_chan, kernel_size=3, padding=1), # 7，3
            nn.GroupNorm(32, n_chan),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(32, n_chan),
            nn.SiLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1),
            nn.GroupNorm(32, n_chan),
            nn.SiLU()
        )


        self.topblock = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.trainable = [self.input_fc, 
                          self.block1, 
                          self.block2, 
                          self.resblocks3, self.att2, self.resblocks4, 
                          self.block3, 
                          self.topblock]

    def forward(self, x):
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        # for block_index, block in enumerate(self.blocks):
        #     x = block(x)
        #     import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.block1(x)

        # x = self.resblocks1(x)
        # x = self.att1(x)
        # x = self.resblocks2(x)

        x = F.interpolate(x, scale_factor=4, mode='bicubic')
        x = self.block2(x)

        x = self.resblocks3(x)
        x = self.att2(x)
        x = self.resblocks4(x)

        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.block3(x)

        # x = self.resblocks5(x)
        # x = self.att3(x)
        # x = self.resblocks6(x)

        x = self.topblock(x)
        
        # x = self.top(x)
        # save_image(x, f'top.png')
        # import pdb;pdb.set_trace()

        return x

class DepthExtractor(nn.Module):
    def __init__(self, img_xfm_norm=identity, model_type=None):
        super(DepthExtractor, self).__init__()
        if not model_type:
            model_type = FLAGS.midas_type
        if model_type == "large":
            model_path = f'{PROJECT_ROOT}/data/model-f6b98070.pt'
            self.model = src.midas.midas_net.MidasNet(model_path, non_negative=True)
            self.net_input_size = 384
        elif model_type == "small":
            model_path = f'{PROJECT_ROOT}/data/model-small-70d6b9c8.pt'
            self.model = src.midas.midas_net_custom.MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            self.net_input_size = 256
        self.model.eval()
        self.img_xfm_norm = img_xfm_norm

    def forward(self, x):
        orig_size = x.shape[-1]
        x = interpolate(x, size=self.net_input_size, mode='bicubic')
        x = self.img_xfm_norm(x)
        pred = self.model.forward(x)
        pred = interpolate(pred.unsqueeze(1), size=orig_size, mode='bicubic').squeeze(1)
        # Normalize
        pred = (pred - pred.view(len(pred), -1).mean(1)[:, None, None]) / (pred.view(len(pred), -1).std(1)[:, None, None] + 1e-4)
        return pred  # NxHxW

def make_model(model_type, *args, **kwargs):
    return globals()[model_type](*args, **kwargs)


# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class fMRIEncoder(nn.Module):
    def __init__(self, n_voxels, ch=32, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[16,8], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=112, z_channels=8, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_CHW = (z_channels,14,14)
        self.n_voxels = n_voxels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.out_fc = nn.Linear(np.prod(self.out_CHW), self.n_voxels)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.view(h.shape[0], -1)
        fmri_pred = self.out_fc(h)
        return fmri_pred


class fMRIDecoder(nn.Module):
    def __init__(self, n_voxels, ch=32, out_ch=3,  num_res_blocks=2,
                 resolution=112, z_channels=8, attn_resolutions=[16, 8],
                 ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.start_CHW = (8, 14, 14)
        self.input_fc = nn.Linear(n_voxels, np.prod(self.start_CHW))

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape
        z = self.input_fc(z)
        z = z.view(-1, *self.start_CHW)

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h




