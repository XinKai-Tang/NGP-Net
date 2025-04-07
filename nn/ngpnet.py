import torch

from torch import nn, Tensor
from torch.nn import functional as F

from .net_utils import *


class NGPnet(nn.Module):
    ''' Nodule Growth Prediction network (NGPnet) '''

    def __init__(self,
                 img_size: int = 64,
                 feat_dim: int = 8,
                 num_classes: int = 2,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 drop_path_rate: float = 0.0, 
                 spatial_dim: int = 3):
        ''' Args:
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `num_classes: number of classes.
        * `depths`: number of blocks in each stage.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(NGPnet, self).__init__()

        self.encoder = UNetEncoder(in_channels=2, 
                                   depths=depths,
                                   img_size=img_size,
                                   feat_dim=feat_dim,
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        
        self.shape_net = UNetDecoder(out_channels=num_classes,
                                     depths=depths,
                                     img_size=img_size,
                                     feat_dim=feat_dim,
                                     drop_path_rate=drop_path_rate,
                                     spatial_dim=spatial_dim)

        self.texture_net = UNetDecoder(out_channels=spatial_dim,
                                       depths=depths,
                                       img_size=img_size,
                                       feat_dim=feat_dim,
                                       drop_path_rate=drop_path_rate,
                                       spatial_dim=spatial_dim)
        
        self.sp_trans = SpatialTransformer(in_size=img_size,
                                           spatial_dim=spatial_dim)

    def ngpnet(self, im0: Tensor, im1: Tensor, tm0: Tensor, tm1: Tensor):
        im = torch.cat([im0, im1], dim=1)
        enc = self.encoder(im, tm0)

        mk2 = self.shape_net(enc, tm1)
        field = self.texture_net(enc, tm1)
        im2 = self.sp_trans(im1, field)
        return im2, mk2, field

    def forward(self, im0: Tensor, im1: Tensor, tm0: Tensor, tm1: Tensor):
        im2, mk2, _ = self.ngpnet(im0, im1, tm0, tm1)
        mk2 = mk2.argmax(dim=1, keepdim=True)
        return im2, mk2


class UNetEncoder(nn.Module):
    ''' Encoder for U-Net '''

    def __init__(self, 
                 in_channels: int = 2,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 img_size: int = 64,
                 feat_dim: int = 8,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_channels`: dimension of input channels.
        * `depths`: number of blocks in each stage.
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(UNetEncoder, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.tokenizer = nn.Sequential(
            DownBlock(in_channels, feat_dim, spatial_dim=spatial_dim),
        )

        self.encoder1 = BlockGroup(feat_dim=feat_dim,
                                   img_size=img_size//2,
                                   num_blocks=depths[0],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler1 = DownBlock(in_channels=feat_dim,
                                      out_channels=feat_dim*2,
                                      spatial_dim=spatial_dim)
        
        self.encoder2 = BlockGroup(feat_dim=feat_dim*2,
                                   img_size=img_size//4,
                                   num_blocks=depths[1],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler2 = DownBlock(in_channels=feat_dim*2,
                                      out_channels=feat_dim*4,
                                      spatial_dim=spatial_dim)
        
        self.encoder3 = BlockGroup(feat_dim=feat_dim*4,
                                   img_size=img_size//8,
                                   num_blocks=depths[2],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)
        self.downsampler3 = DownBlock(in_channels=feat_dim*4,
                                      out_channels=feat_dim*8,
                                      spatial_dim=spatial_dim)

    def forward(self, xs: Tensor, t0: Tensor):
        e0 = self.tokenizer(xs)
        e1 = self.downsampler1(self.encoder1(e0, t0))
        e2 = self.downsampler2(self.encoder2(e1, t0))
        e3 = self.downsampler3(self.encoder3(e2, t0))
        return (e0, e1, e2, e3)


class UNetDecoder(nn.Module):
    ''' Decoder for U-Net '''

    def __init__(self, 
                 out_channels: int = 3,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 img_size: int = 64,
                 feat_dim: int = 8,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `out_channels`: dimension of output channels.
        * `depths`: number of blocks in each stage.
        * `img_size`: spatial size of input images.
        * `feat_dim: output channels of the tokenizer.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(UNetDecoder, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.center = BlockGroup(feat_dim=feat_dim*8,
                                 img_size=img_size//16,
                                 num_blocks=depths[3],
                                 drop_path_rate=drop_path_rate,
                                 spatial_dim=spatial_dim)
        
        self.upsampler3 = UpBlock(in_channels=feat_dim*8,
                                  out_channels=feat_dim*4,
                                  spatial_dim=spatial_dim)
        self.decoder3 = BlockGroup(feat_dim=feat_dim*4,
                                   img_size=img_size//8,
                                   num_blocks=depths[2],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.upsampler2 = UpBlock(in_channels=feat_dim*4,
                                  out_channels=feat_dim*2,
                                  spatial_dim=spatial_dim)
        self.decoder2 = BlockGroup(feat_dim=feat_dim*2,
                                   img_size=img_size//4,
                                   num_blocks=depths[1],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.upsampler1 = UpBlock(in_channels=feat_dim*2,
                                  out_channels=feat_dim,
                                  spatial_dim=spatial_dim)
        self.decoder1 = BlockGroup(feat_dim=feat_dim,
                                   img_size=img_size//2,
                                   num_blocks=depths[0],
                                   drop_path_rate=drop_path_rate,
                                   spatial_dim=spatial_dim)

        self.out_head = nn.Sequential(
            UpBlock(feat_dim, feat_dim, spatial_dim=spatial_dim),
            nn.GroupNorm(num_groups=feat_dim//4, num_channels=feat_dim),
            Conv(feat_dim, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, xs: Tensor, t1: Tensor):
        (e0, e1, e2, e3) = xs
        d3 = self.center(e3, t1)
        d2 = self.decoder3(self.upsampler3(e3 + d3), t1)
        d1 = self.decoder2(self.upsampler2(e2 + d2), t1)
        d0 = self.decoder1(self.upsampler1(e1 + d1), t1)
        out = self.out_head(e0 + d0)
        return out


class SpatialTransformer(nn.Module):
    ''' Spatial Transformer Network (STN) '''

    def __init__(self,
                 in_size: int = 64,
                 mode_img: str = "bilinear",
                 mode_msk: str = "nearest",
                 spatial_dim: int = 3):
        ''' Args:
        * `in_size`: spatial size of input images.
        * `mode_img`: sample mode for images.
        * `mode_msk`: sample mode for masks.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(SpatialTransformer, self).__init__()
        self.size = [in_size,] * spatial_dim
        self.mode_img = mode_img
        self.mode_msk = mode_msk
        
        grid = torch.stack(torch.meshgrid([
            torch.arange(0, s) for s in self.size
        ], indexing="ij")).float().unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self, src: Tensor, flow: Tensor, is_msk: bool = False):
        # normalize grid values to [-1,1] for resampler
        loc = self.grid + flow
        for i, s in enumerate(self.size):
            loc[:, i] = 2 * (loc[:, i] / (s - 1) - 0.5)

        # move channels dim to last position
        if len(self.size) == 2:
            loc = loc.permute(0, 2, 3, 1)[..., [1, 0]]
        else:
            loc = loc.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        mode = self.mode_msk if is_msk else self.mode_img
        dst = F.grid_sample(src, loc, mode=mode, align_corners=True)
        return dst


if __name__ == '__main__':
    INTERVAL, IMG_SIZE = 64, (1, 1, 48, 48, 48)
    model = NGPnet(img_size=IMG_SIZE[-1], feat_dim=8)   # 1.32 M
    im0 = torch.randn(size=IMG_SIZE)
    im1 = torch.randn(size=IMG_SIZE)
    tm0 = torch.randint(0, INTERVAL, size=(IMG_SIZE[0],))
    tm1 = torch.randint(0, INTERVAL, size=(IMG_SIZE[0],))
    im1, mk1 = model(im0, im1, tm0, tm1)
    n_params = sum([param.nelement() for param in model.parameters()])
    print(model)
    print("Dimension of outputs:", im1.shape, mk1.shape)
    print("Number of parameters: %.2fM" % (n_params / 1024**2))
