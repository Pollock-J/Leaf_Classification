# https://github.com/tyeso/Image_Classification_with_CoAtNet_and_ResNet18
import math
import torch
import torch.nn as nn
import torch.nn.intrinsic
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd


class SELayer(nn.Module):
    """
    This class is adapted from the efficientnetv2 repo.
    https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, SELayer._make_divisible(inp // reduction, 8)),
                nn.SiLU(),
                nn.Linear(SELayer._make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


class MBConv(nn.Module):
    """
    This class is adapted from the efficientnetv2 repo.
    https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            SELayer(inp, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SDPAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(SDPAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, mask=None):
        atten = torch.bmm(q, k.transpose(1, 2))
        if scale:
            atten = atten * scale
        if mask:
            atten = atten.masked_fill_(mask, -np.inf)
        atten = self.softmax(atten)
        atten = self.dropout(atten)
        output = torch.bmm(atten, v)
        return output, atten


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_ch, out_chs):
        super().__init__()
        self.s0 = nn.Sequential(nn.Conv3d(), nn.Conv3d(), MBConv())
        self.c = MBConv()
        self.t = nn.Sequential(SDPAttention(), nn.Linear(), nn.ReLU(), nn.Linear())

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.s0(x)
        y = self.maxpool2d(y)
        y = self.c()
        y = rearrange(y, 'BCHW -> BHWC')
        y = y.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)  # B,N,C
        y = self.t(y,y,y)
        y = self.maxpool1d(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.t(y,y,y)
        y = self.maxpool1d(y.permute(0, 2, 1))
        n = y.shape[-1]
        y = y.reshape(B, self.out_chs[4], int(math.sqrt(n)), int(math.sqrt(n)))
        return y


if __name__ == '__main__':
    # File paths for the csv which contain the picture path, plant type, and if it was augmented
    csvs = ('100 leaves/pic_data.csv', 'leafsnap-dataset/pic_data.csv', 'swedish leaves/pic_data.csv')
    # Pulling the information from the csvs
    for csv in csvs:
        df = pd.read_csv(csv)
        path = df.columns[0]
        classification = df.columns[2]
        full_path = full_path.append(path)
        full_classification = full_classification.append(classification)
        train_size = int(0.6 * len(df))
        test_size = int(0.2 * len(df))
        validation = int(0.2 * len(df))
        train_dataset, validation, test_dataset = torch.utils.data.random_split(df, [train_size, validation, test_size])

    coatnet = CoAtNet(3,256)