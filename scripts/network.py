import torch
import torch.nn as nn
import torch.nn.functional as F


#ACTIVATION = nn.ReLU
ACTIVATION = nn.SELU


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, do_pooling):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) if do_pooling else None
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, dilation=2, bias=True, padding='same'),
            ACTIVATION(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, dilation=2, bias=True, padding='same'),
            ACTIVATION(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.dropout = nn.AlphaDropout(p=0.5)
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, bias=True)
        self.deconv_sigma = ACTIVATION()
        self.deconv_bn = nn.BatchNorm2d(in_channels)
        self.skip_block = nn.Sequential(
            nn.Conv2d(skip_channels, in_channels, kernel_size=1, bias=True, padding='same'),
            ACTIVATION(),
            nn.BatchNorm2d(in_channels),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, bias=True, padding='same'),
            ACTIVATION(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, bias=True, padding='same'),
            ACTIVATION(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, skip):
        x = self.dropout(self.deconv_bn(self.deconv_sigma(self.deconv(x, output_size=skip.size()))))
        return self.conv_block(x + self.skip_block(skip))


# class UNetlikeNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = ConvBlock(1, 32, False)
#         self.conv2 = ConvBlock(32, 32, True)
#         self.conv3 = ConvBlock(32, 32, True)
#         self.conv4 = ConvBlock(32, 32, True)
#         self.deconv1 = DeconvBlock(32, 32, 32)
#         self.deconv2 = DeconvBlock(32, 32, 16)
#         self.deconv3 = DeconvBlock(16, 32, 8)
#         self.class_output = nn.Sequential(
#             nn.Conv2d(8, 1, kernel_size=1, bias=True, padding='same'),
#             nn.Sigmoid(),
#         )
#         self.flow_output = nn.Sequential(
#             nn.Conv2d(8, 2, kernel_size=1, bias=True, padding='same'),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x = self.conv4(x3)
#         x = self.deconv1(x, x3)
#         x = self.deconv2(x, x2)
#         x = self.deconv3(x, x1)
#         return self.class_output(x), self.flow_output(x)


class UNetlikeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 32, False)
        self.conv2 = ConvBlock(32, 32, True)
        self.conv3 = ConvBlock(32, 32, True)
        self.conv4 = ConvBlock(32, 64, True)
        self.conv5 = ConvBlock(64, 64, True)
        self.deconv1 = DeconvBlock(64, 64, 64)
        self.deconv2 = DeconvBlock(64, 32, 32)
        self.deconv3 = DeconvBlock(32, 32, 16)
        self.deconv4 = DeconvBlock(16, 32, 8)
        self.class_output = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1, bias=True, padding='same'),
            nn.Sigmoid(),
        )
        self.flow_output = nn.Sequential(
            nn.Conv2d(8, 2, kernel_size=1, bias=True, padding='same'),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv5(x4)
        x = self.deconv1(x, x4)
        x = self.deconv2(x, x3)
        x = self.deconv3(x, x2)
        x = self.deconv4(x, x1)
        return self.class_output(x), self.flow_output(x)
