import math
import torch

from typing import Sequence, Union
from torch import nn, Tensor
from torch.nn import functional as F


def layer_norm(x: Tensor,
               norm_shape: Sequence[int],
               eps: float = 1e-6,
               channels_last: bool = True):
    if channels_last:   # [B, ..., C]
        y = F.layer_norm(x, norm_shape, eps=eps)
    else:               # [B, C, ...]
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
    return y


class DropPath(nn.Module):
    ''' Stochastic drop paths per sample for residual blocks.

        Reference: https://github.com/rwightman/pytorch-image-models
    '''

    def __init__(self,
                 drop_prob: float = 0.0,
                 scale_by_keep: bool = True):
        ''' Args:
            * `drop_prob`: drop paths probability.
            * `scale_by_keep`: whether scaling by non-dropped probaility.
        '''
        super(DropPath, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError("drop_path_prob should be between 0 and 1.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor):
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            rand_tensor.div_(keep_prob)
        return x * rand_tensor


class LayerNorm(nn.Module):
    ''' Layer Normalization (LN) '''

    def __init__(self,
                 norm_shape: Union[Sequence[int], int],
                 eps: float = 1e-6,
                 channels_last: bool = True):
        ''' Args:
        * `norm_shape`: dimension of the input feature.
        * `eps`: epsilon of layer normalization.
        * `channels_last`: whether the channel is the last dim.
        '''
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(norm_shape), requires_grad=True)
        self.channels_last = channels_last
        self.norm_shape = (norm_shape,)
        self.eps = eps

    def forward(self, x: Tensor):
        if self.channels_last:  # [B, ..., C]
            y = F.layer_norm(x, self.norm_shape,
                             self.weight, self.bias, self.eps)
        else:                   # [B, C, ...]
            y = layer_norm(x, self.norm_shape, self.eps, False)
            if x.ndim == 4:
                y = self.weight[:, None, None] * y
                y += self.bias[:, None, None]
            else:
                y = self.weight[:, None, None, None] * y
                y += self.bias[:, None, None, None]
        return y


class GlobalRespNorm(nn.Module):
    ''' Global Response Normalization (GRN) '''

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 channels_last: bool = False,
                 spatial_dim: int = 3):
        ''' Args:
        * `dim`: dimension of input channels.
        * `eps`: epsilon of the normalization.
        * `channels_last`: whether the channel is the last dim.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(GlobalRespNorm, self).__init__()
        if spatial_dim == 2:
            if channels_last:
                size, self.dims = [1, 1, 1, dim], (1, 2)
            else:
                size, self.dims = [1, dim, 1, 1], (2, 3)
        else:
            if channels_last:
                size, self.dims = [1, 1, 1, 1, dim], (1, 2, 3)
            else:
                size, self.dims = [1, dim, 1, 1, 1], (2, 3, 4)

        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(*size))
        self.beta = nn.Parameter(torch.zeros(*size))

    def forward(self, x: Tensor):
        ''' Args:
        * `x`: a input tensor in [B, C, (D,) W, H] shape.
        '''
        gx = torch.norm(x, p=2, dim=self.dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


class TEM(nn.Module):
    ''' Time Embedding Module (TEM) '''

    def __init__(self,
                 time_dim: int = 8,
                 img_size: int = 32,
                 spatial_dim: int = 3):
        ''' Args:
        * `time_dim`: dimension of time embedding.
        * `img_size`: size of feature map.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(TEM, self).__init__()
        self.shape = [time_dim] + [1] * spatial_dim

        half = (img_size ** spatial_dim) // 2
        scale = -math.log(1e4) / (half - 1)
        self._2i = (scale * torch.arange(half)).exp()

        self.linear = nn.Sequential(
            nn.Linear(2 * half, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: Tensor):
        if self._2i.device != t.device:
            self._2i = self._2i.to(t.device)
        itv = t[:, None] * self._2i[None, :]
        emb = torch.cat((itv.sin(), itv.cos()), dim=-1)
        emb = self.linear(emb).view(t.shape[0], *self.shape)
        return emb


class STEM(nn.Module):
    ''' Spatial-Temporal Encoding Module (STEM) '''

    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(STEM, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        ### spatial embedding:
        self.norm = LayerNorm(in_dim, eps=1e-6, channels_last=False)
        self.qk_sp = Conv(in_dim, in_dim, 1, groups=in_dim)
        self.v_sp = Conv(in_dim, in_dim, 1, groups=in_dim)
        ### temporal embedding:
        self.qk_tm = Conv(in_dim, in_dim, 1, groups=in_dim)
        self.v_tm = Conv(in_dim, in_dim, 1, groups=in_dim)
        ### attention:
        self.swish = nn.SiLU(inplace=True)
        self.attn = Conv(in_dim, in_dim, 4, padding=3, groups=in_dim, dilation=2)
        ### projection:
        self.proj = nn.Sequential(
            GlobalRespNorm(in_dim, spatial_dim=spatial_dim, 
                           eps=1e-6, channels_last=False),
            Conv(in_dim, in_dim, 1, groups=in_dim),
            DropPath(drop_path_rate, scale_by_keep=True),
        )

    def forward(self, xs: Tensor, xt: Tensor):
        x0, xs = xs, self.norm(xs)
        ### query,key,value:
        qk = self.qk_sp(xs) + self.qk_tm(xt)
        v = self.v_sp(xs) + self.v_tm(xt)
        ### attention:
        x = self.attn(self.swish(qk)) * v
        ### projection:
        y = x0 + self.proj(x)
        return y


class ReSTEBlock(nn.Module):
    ''' Residual Spatial-Temporal Encoding Block (ReSTE block) '''
    
    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(ReSTEBlock, self).__init__()
        Conv = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        self.conv = Conv(in_dim, in_dim, 3, padding=1)
        self.swish = nn.SiLU(inplace=True)
        self.norm = nn.GroupNorm(num_groups=in_dim//4,
                                 num_channels=in_dim)
        self.block = STEM(in_dim=in_dim,
                          spatial_dim=spatial_dim,
                          drop_path_rate=drop_path_rate)
        
    def forward(self, x: Tensor, t: Tensor):
        y = self.conv(self.swish(self.norm(x)))
        y = x + self.block(y, t)
        return y


class BlockGroup(nn.Module):
    ''' ReSTE Block Group '''

    def __init__(self,
                 feat_dim: int = 16,
                 img_size: int = 64,
                 num_blocks: int = 2,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `feat_dim`: channels of image features.
        * `img_size`: size of image features.
        * `num_blocks`: number of blocks in each stage.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(BlockGroup, self).__init__()
        
        self.time_emb = TEM(time_dim=feat_dim,
                            img_size=img_size,
                            spatial_dim=spatial_dim)
        self.blocks = nn.ModuleList([
            ReSTEBlock(in_dim=feat_dim,
                       drop_path_rate=drop_path_rate,
                       spatial_dim=spatial_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, xs: Tensor, t: Tensor):
        xt = self.time_emb(t)
        for block in self.blocks:
            xs = block(xs, xt)
        return xs


class DownBlock(nn.Module):
    ''' Downsampling Block '''

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 2, 
                 padding: int = 1,
                 spatial_dim: int = 3):
        super(DownBlock, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: Tensor):
        return self.conv(x)


class UpBlock(nn.Module):
    ''' Upsampling Block '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 output_padding: int = 1,
                 spatial_dim: int = 3):
        super(UpBlock, self).__init__()
        if spatial_dim in [2, 3]:
            ConvT = nn.ConvTranspose3d if spatial_dim == 3 else nn.ConvTranspose2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.conv = ConvT(in_channels, out_channels, kernel_size, 
                          stride, padding, output_padding)

    def forward(self, x: Tensor):
        return self.conv(x)
