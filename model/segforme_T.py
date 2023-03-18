from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

#上采样
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# 卷积层
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

# 双卷积
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
# classes
class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            #(2,3,512,512)
            x = get_overlap_patches(x)
            #(2,147,16384)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            #(2,147,128,128)

            x = overlap_embed(x)
            #(2,32,128,128)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer_primary(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 2
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        # 原始特征融合方法
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        # 原始分类方法
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
            ##(2,2,128,128)
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
    def forward(self, x):           
        b,n,h,w = x.shape
        #(2,3,512,512)
        layer_outputs = self.mit(x, return_layer_outputs = True)
        #(2,32,128,128),(2,64,64,64),(2,160,32,32),(2,256,16,16)

        # 原始的方法
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        (2,1024,128,128)
        output = self.to_segmentation(fused)
        (2,2,512,512)
        return output

class Segformer_upsample(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 2
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        # 原始特征融合方法
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        # 改方法1(亚像素)
        self.to_segmentation1 = nn.Sequential(
            nn.PixelShuffle(4),
            nn.Conv2d(decoder_dim//4, num_classes, 1),
        )
        # 改方法2(反卷积)
        self.to_segmentation2 = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            #(1,256,128,128)
            nn.ConvTranspose2d(decoder_dim,decoder_dim//2, kernel_size=2, stride=2),
            #(1,128,256,256)
            nn.ConvTranspose2d(decoder_dim//2,num_classes, kernel_size=2, stride=2),
            #(1,2,512,512)
        )
    def forward(self, x):           
        b,n,h,w = x.shape
        #(2,3,512,512)
        layer_outputs = self.mit(x, return_layer_outputs = True)
        #(2,32,128,128),(2,64,64,64),(2,160,32,32),(2,256,16,16)

        # 原始的方法
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        (2,1024,128,128)
        output = self.to_segmentation2(fused)
        (2,2,512,512)
        return output

class Segformer_unet(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 2
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        # 改进类似unet解码结构 method1
        # self.dc1 = DoubleConv(dims[-1],dims[-1])
        # self.up1 = up_conv(dims[-1],dims[-2])
        # self.dc2 = DoubleConv(dims[-2]*2,dims[-2])
        # self.up2 = up_conv(dims[-2],dims[-3])
        # self.dc3 = DoubleConv(dims[-3]*2,dims[-3])
        # self.up3 = up_conv(dims[-3],dims[0])
        # self.dc4 = DoubleConv(dims[0]*2,dims[0])
        # self.to_segmentation = nn.Sequential(
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #     nn.Conv2d(dims[0], num_classes, 1),
        # )


        ## 改进类似unet解码结构（增加低维卷积特征提取）  method2  
        self.low_conv = SingleConv(3, dims[0], kernel_size=7, stride=2, padding=3)
        self.dc1 = DoubleConv(dims[-1],dims[-1])
        self.up1 = up_conv(dims[-1],dims[-2])
        self.dc2 = DoubleConv(dims[-2]*2,dims[-2])
        self.up2 = up_conv(dims[-2],dims[-3])
        self.dc3 = DoubleConv(dims[-3]*2,dims[-3])
        self.up3 = up_conv(dims[-3],dims[0])
        self.dc4 = DoubleConv(dims[0]*2,dims[0])
        self.to_segmentation = nn.Sequential(
            # nn.Conv2d(dims[0]*2,dims[0]*2,kernel_size=1),
            # nn.BatchNorm2d(dims[0]*2),
            # nn.ReLU(inplace=True),
            SingleConv(dims[0]*2,dims[0]*2,kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[0]*2, num_classes, 1))

    def forward(self, x):           
        b,n,h,w = x.shape
        #(2,3,512,512)
        layer_outputs = self.mit(x, return_layer_outputs = True)
        #(2,32,128,128),(2,64,64,64),(2,160,32,32),(2,256,16,16)

        # # methond1
        # x_c = self.dc1(layer_outputs[-1])
        # x_c = self.up1(x_c)
        # x_c = self.dc2(torch.cat([x_c,layer_outputs[-2]], dim=1))
        # x_c = self.up2(x_c)
        # x_c = self.dc3(torch.cat([x_c,layer_outputs[-3]], dim=1))
        # x_c = self.up3(x_c)
        # x_c = self.dc4(torch.cat([x_c,layer_outputs[0]], dim=1))
        # # (2,32,128,128)
        # output = self.to_segmentation(x_c)

        ## methond2 增加低维卷积特征提取
        x_l= self.low_conv(x)
        # (1,32,256,256)
        x_c = self.dc1(layer_outputs[-1])
        x_c = self.up1(x_c)
        x_c = self.dc2(torch.cat([x_c,layer_outputs[-2]], dim=1))
        x_c = self.up2(x_c)
        x_c = self.dc3(torch.cat([x_c,layer_outputs[-3]], dim=1))
        x_c = self.up3(x_c)
        x_c = self.dc4(torch.cat([x_c,layer_outputs[0]], dim=1))
        # (2,32,128,128)
        output = self.to_segmentation(torch.cat([F.interpolate(x_c,scale_factor=2),x_l],dim=1))

        return output

class Segformer_deeplabv3plus(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 2
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        ## 改进类似deeplabv3plus解码结构（增加低维卷积特征提取）
        self.low_conv = SingleConv(3, dims[1], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.conv1 = SingleConv(dims[-1]*4,dims[-1],1)

        self.to_segmentation = nn.Sequential(
            SingleConv((decoder_dim+dims[1]), decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
            ##(2,2,128,128)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):

        # 增加低维卷积特征提取
        x_l= self.low_conv(x)
        # (1,64,256,256)

        layer_outputs = self.mit(x, return_layer_outputs = True)
        #(2,32,128,128),(2,64,64,64),(2,160,32,32),(2,256,16,16)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        # (2,1024,128,128)

        fused = F.interpolate(self.conv1(fused),scale_factor=2,mode="bilinear")
        # (2,256,256,256)

        fused = torch.cat([fused,x_l], dim = 1)
        # (2,320,256,256)
        output = self.to_segmentation(fused)
        return output


def m_segformer_T(num_classes=2):
    '''
    调用
    '''
    model = Segformer_deeplabv3plus(
    dims = (32, 64, 160, 256),      # dimensions of each stage
    heads = (1, 2, 5, 8),           # heads of each stage
    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    num_layers = 2,                 # num layers of each stage
    decoder_dim = 256,              # decoder dimension
    num_classes = num_classes                 # number of segmentation classes
    )
    return model
    
if __name__ =="__main__":
    model = m_segformer_T()
    x = torch.randn(2, 3, 512, 512)
    pred = model(x)
    print(pred.shape)
    # print(model)
    # y = mit(x)
    # print(y.shape)
    