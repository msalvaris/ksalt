import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class SpatialGate2d(nn.Module):
    def __init__(self, channels):
        super(SpatialGate2d, self).__init__()
        self.spatial_se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spa_se = self.spatial_se(x)
        return spa_se


class ChannelGate2d(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(
            nn.Linear(channels, int(channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        return chn_se


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = SpatialGate2d(out_channels)
        self.channel_gate = ChannelGate2d(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode="bilinear", align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 * x + g2 * x
        return x


class UNetResNetSCSE(nn.Module):
    def __init__(self):
        super(UNetResNetSCSE, self).__init__()
        # self.resnet = ReSnet(BasicBlock, [3, 4, 6, 3], num_classes=1)
        self.resnet = resnet34(pretrained=True)
        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)

        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat(
            [(x - mean[2]) / std[2], (x - mean[1]) / std[1], (x - mean[0]) / std[0]], 1
        )

        x = self.conv1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder2(e2)
        e4 = self.encoder2(e3)
        e5 = self.encoder2(e4)

        f = self.center(e5)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        f = torch.cat(
            (
                d1,
                F.upsample(d2, scale_factor=2, mode="bilinear", align_corners=False),
                F.upsample(d3, scale_factor=4, mode="bilinear", align_corners=False),
                F.upsample(d4, scale_factor=8, mode="bilinear", align_corners=False),
                F.upsample(d5, scale_factor=16, mode="bilinear", align_corners=False),
            ),
            1,
        )

        f = F.dropout2d(f, p=0.5)
        logit = self.logit(f)
        return logit
