import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,kernel_size = 3, pool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding=1)
        self.norm1 =nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size= kernel_size, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, input_image):
        x_1 = self.activation(self.norm1(self.conv1(input_image)))
        x_2 = self.activation(self.norm1(self.conv2(x_1)))
        feature_map = self.pool(x_2)

        return feature_map

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4):
        super().__init__()
        
        self.convblocks =  nn.ModuleList()
        current_in = in_channels
        current_out = base_channels

        for level in range(levels):
            is_bottom = (level == (levels -1))
            self.convblocks.append(ConvBlock(in_channels=current_in, out_channels=current_out, pool = not is_bottom))
            current_in = current_out
            current_out = 2* current_in

        self.out_channels = current_in
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)

    def forward(self, input_image):
        x = input_image
        for block in self.convblocks:
            x = block(x)

        x_1 = self.gap(x)
        x_1 = x_1.flatten(1)
        return x_1

class FC_Block(nn.Module):
    def __init__(self, input_units = 512, output_units = 8, depth = 2, dropout = 0.4):
        super().__init__()

        layers = [nn.Dropout(dropout)]
        current_units = input_units

        for _ in range(depth - 1):
            hidden_units = max(input_units // 8 , output_units)
            layers += [nn.Linear(in_features=current_units, out_features= hidden_units),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout)]
            current_units = hidden_units 
        layers.append(nn.Linear(in_features= current_units, out_features= output_units))

        self.network = nn.Sequential(*layers)
        
        

    def forward(self, features):
        return self.network(features)


class NodeConvs_Net(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4, dropout = 0.4, fc_depth = 2, num_classes = 8):
        super().__init__()

        self.encoder = Encoder(in_channels= in_channels, base_channels= base_channels, levels= levels)
        self.Fc = FC_Block(input_units=self.encoder.out_channels, depth=fc_depth, output_units=num_classes, dropout=dropout)

    def forward(self,input_image):
        feature_tensor = self.encoder(input_image)
        output_logits = self.Fc(feature_tensor)

        return output_logits

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Sanity check
if __name__ == "__main__":
    model = NodeConvs_Net(in_channels=3, base_channels=64, levels=4, num_classes=8)
    dummy_batch = torch.randn(4, 3, 64, 64)  # batch of 4 RGB 64x64 images
 
    logits = model(dummy_batch)
    print(logits.shape)  # expected: torch.Size([4, 8])
    print(f"Total trainable parameters: {model.count_trainable_parameters():,}")

