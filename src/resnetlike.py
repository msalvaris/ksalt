"""
Contains UNet implementation using a combination of ResNet and ResNext layer methods
"""

from torch.nn import init
import math
from toolz import pipe

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()
        
class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(inplanes, D * cardinality, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * cardinality)

        self.conv_conv = nn.Conv2d(D * cardinality, D * cardinality, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                                   bias=False)
        self.bn = nn.BatchNorm2d(D * cardinality)

        self.conv_expand = nn.Conv2d(D * cardinality, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)
    

class PreactivationResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreactivationResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d()
        # relu
        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d()
        # relu
        # Conv block 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        
        return x+residual
        

class EncodingLayer(nn.Module):
    def __init__(self, channels, dropout_p=0.5):
        super(EncodingLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1 = PreactivationResidualBlock(channels, channels)
        self.res2 = PreactivationResidualBlock(channels, channels)
        self.bn = nn.BatchNorm2d()
        # relu
        self.max = nn.MaxPool2D(2)
        self.drop = nn.Dropout2d(p=dropout_p)
    
    def forward(self, x):
        return pipe(x,
                    self.conv,
                    self.res1,
                    self.res2,
                    self.bn,
                    F.relu,
                    self.max,
                    self.drop)


class ResidualLayer(nn.Module):
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1 = PreactivationResidualBlock(channels, channels)
        self.res2 = PreactivationResidualBlock(channels, channels)
        self.bn = nn.BatchNorm2d()
        # relu
    
    def forward(self, x):
        return pipe(x,
                    self.conv,
                    self.res1,
                    self.res2,
                    self.bn,
                    F.relu)
    

class DropResidualLayer(nn.Module):
    def __init__(self, channels, dropout_p=0.5):
        super(DropResidualLayer, self).__init__()
        self.drop = nn.Dropout2d(p=dropout_p)
        self.res = ResidualLayer(channels)
    
    def forward(self, x):
        return pipe(x,
             self.drop,
             self.res)
        
    
class UNetResNet(nn.Module):
    def __init__(self, channels, dropout_p=0.5):
        super(UNetResNet, self).__init__()

        self.enc1 = EncodingLayer(channels)
        self.enc2 = EncodingLayer(channels*2)
        self.enc3 = EncodingLayer(channels*4)
        self.enc4 = EncodingLayer(channels*8)
        self.middle = ResidualLayer(channels*16)
        self.dec4 = nn.ConvTranspose2d(channels*16, channels*8,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           output_padding=0)
        #concat dec4 + enc4
        self.drop_res4 = DropResidualLayer(channels*8, dropout_p=dropout_p/2)

        self.dec3 = nn.ConvTranspose2d(channels * 8, channels * 4,
                                       kernel_size=3,
                                       stride=2,
                                       padding=0,
                                       output_padding=0)
        # concat dec3 + enc3
        self.drop_res3 = DropResidualLayer(channels * 4, dropout_p=dropout_p)

        self.dec2 = nn.ConvTranspose2d(channels * 4, channels * 2,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=0)
        # concat dec2 + enc2
        self.drop_res2 = DropResidualLayer(channels * 2, dropout_p=dropout_p)

        self.dec1 = nn.ConvTranspose2d(channels * 2, channels,
                                       kernel_size=3,
                                       stride=2,
                                       padding=0,
                                       output_padding=0)
        # concat dec1 + enc1
        self.drop_res1 = DropResidualLayer(channels, dropout_p=dropout_p)

        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(initialize_weights)
    
    
def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc3(enc2)
    enc4 = self.enc4(enc3)
    middle = self.middle(enc4)
    dec4 = self.dec4(middle)
    drop_res4 = self.drop_res4(torch.cat([dec4, enc4], 1))
    dec3 = self.dec4(drop_res4)
    drop_res3 = self.drop_res3(torch.cat([dec3, enc3], 1))
    dec2 = self.dec4(drop_res3)
    drop_res2 = self.drop_res4(torch.cat([dec2, enc2], 1))
    dec1 = self.dec4(drop_res2)
    drop_res1 = self.drop_res4(torch.cat([dec1, enc1], 1))
    final_conv = self.final_conv(drop_res1)
    return self.sigmoid(final_conv)
    
    


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, block, depth, cardinality, base_width, num_classes):
        super(CifarResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnext29_16_64(num_classes=10):
    """Constructs a ResNeXt-29, 16*64d model for CIFAR-10 (by default)
  
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNeXt(ResNeXtBottleneck, 29, 16, 64, num_classes)
    return model


def resnext29_8_64(num_classes=10):
    """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
  
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes)
    return model
