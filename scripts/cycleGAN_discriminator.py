from __future__ import print_function

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

import requests
from io import BytesIO

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResudalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1,
                      stride=stride, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    features = [64, 128, 256, 512]

    def __init__(self, in_channels):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, self.features[0], kernel_size=4, stride=2,
                      padding=1, padding_mode='reflect'),
            nn.LeakyReLU(.2)
        )

        layers = []
        for ind, feature in enumerate(self.features[1:]):
            layers.append(ResudalBlock(self.features[ind], feature,
                                       stride=2 if ind != len(self.features) - 2 else 1))
        layers.append(nn.Conv2d(self.features[-1], 1, kernel_size=4,
                                stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return F.sigmoid(self.model(x))


class Generator(nn.Module):
    pass

