import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 downsample=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if downsample else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        # self.block = nn.Sequential(
        #     ConvBlock(channels, channels, kernel_size=3, padding=1),
        #     ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        # )
        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )
        # super().__init__()
        #
        # self.initial = nn.Sequential(
        #     nn.Conv2d(img_channels, num_features,
        #               kernel_size=7, padding=3, padding_mode='reflect'),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.down_blocks = nn.ModuleList(
        #     [
        #         ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
        #         ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        #     ]
        # )
        #
        # self.res_blocks = nn.Sequential(
        #     *[ResidualBlock(num_features * 4) for _ in range(num_res)]
        # )
        #
        # self.up_blocks = nn.ModuleList(
        #     [
        #         ConvBlock(num_features * 4, num_features * 2, downsample=False,
        #                   kernel_size=3, stride=2, padding=1, output_padding=1),
        #         ConvBlock(num_features * 2, num_features, downsample=False,
        #                   kernel_size=3, stride=2, padding=1, output_padding=1),
        #     ]
        # )
        #
        # self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        return self.main(x)
