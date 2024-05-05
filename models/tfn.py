import torch.nn as nn
import torch
from torch.nn import functional as F


class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class TFN(nn.Module):
    def __init__(self, num_classes):
        super(TFN, self).__init__()

        self.conv1 = ConvBNSiLU(in_channels=3, out_channels=32, kernel_size=6, stride=2, padding=1)
        self.conv2 = ConvBNSiLU(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.conv3 = ConvBNSiLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBNSiLU(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.conv5 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.conv6 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv7 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBNSiLU(in_channels=128, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv9 = ConvBNSiLU(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv10 = ConvBNSiLU(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv11 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv12 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = ConvBNSiLU(in_channels=128, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv14 = ConvBNSiLU(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv15 = ConvBNSiLU(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv16 = ConvBNSiLU(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv17 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv18 = ConvBNSiLU(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv19 = ConvBNSiLU(in_channels=128, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv20 = ConvBNSiLU(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=1)
        self.conv21 = ConvBNSiLU(in_channels=192, out_channels=256, kernel_size=1, stride=1, padding=1)

        self.conv22 = ConvBNSiLU(in_channels=640, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x1 = self.max_pool2(x)

        x = self.conv8(x1)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x2 = self.max_pool3(x)

        x = self.conv13(x2)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x3 = self.max_pool4(x)

        x = self.conv19(x3)
        x = self.conv20(x)
        x = self.conv21(x)

        size = x.size()[2:]
        x1 = F.interpolate(x1, size=size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x], dim=1)

        x = self.conv22(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return F.softmax(x, dim=1)
