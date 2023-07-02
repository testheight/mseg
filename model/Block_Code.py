import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from timm.models._efficientnet_blocks import DepthwiseSeparableConv

class single_block(nn.Module):
    """
    single Convolution Block                         单层卷积
    """
    def __init__(self, in_ch, out_ch):
        super(single_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x   

class conv_block(nn.Module):
    """
    Dual Convolution Block                           双层卷积
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class dws_conv(nn.Module):
    """
    DepthwiseSeparable Convolution Block        深度可分离卷积
    """    
    def __init__(self, in_ch, out_ch):
        super(dws_conv, self).__init__()

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch,in_ch),
            DepthwiseSeparableConv(in_ch,out_ch)
)

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block                        上采样
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

class CBAMLayer(nn.Module):
    '''
    CBAM                                        CBAM注意力机制
    '''
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class ChannelAttention(nn.Module):
    """
    ChannelAttention Module1                    通道注意力机制
    """
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少,因此需要一个比例系数ratio进行缩放
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class ChannelAttention2(nn.Module):
    """
    ChannelAttention Module2                    通道注意力机制2
    """
    def __init__(self, in_planes, out_planes, ratio=8):
        """
        第一层全连接层神经元个数较少,因此需要一个比例系数ratio进行缩放
        """
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, out_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class SpatialAttention(nn.Module):
    '''
    SpatialAttention                            空间注意力机制
    '''
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
class CBAM(nn.Module):
    '''
    CBAM                                        CBAM注意力机制
    '''
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
 
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
    
class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling              空洞卷积空间金字塔池化模块
    '''
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
        self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
        )
        self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
        self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
            
        self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)  
    def forward(self, x):
        [b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class SpatialTransformer(nn.Module):
    '''
    Spatial Transformer Networks                STN
    '''
    def __init__(self, spatial_dims):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims 
        self.fc1 = nn.Linear(32*4*4, 1024) # 可根据自己的网络参数具体设置
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x): 
        batch_images = x #保存一份原始数据
        x = x.view(-1, 32*4*4)
        # 利用FC结构学习到6个参数
        x = self.fc1(x)
        x = self.fc2(x) 
        x = x.view(-1, 2,3) # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points)
        return rois, affine_grid_points

if __name__ == "__main__":
    net = CBAM(64)
    x =torch.rand(2,64,64,64)
    y = net(x)
    print(y.shape)