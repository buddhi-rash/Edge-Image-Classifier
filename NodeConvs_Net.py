import numpy as np
import torch
import torch.nn as nn
import matplotlib.plypot as plt


class Residual_Block(nn.Module):
    def __init__(self, image_resolution, in_channels, out_channels ,kernal_size = 3, batch_size = 8):
        super.__init__()

        self.conv = nn.conv2d(in_channels = in_channels, out_channels = out_channels, kernal_size = kernal_size)
        self.norm =nn.BatchNorm2d()

