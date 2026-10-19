import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from NodeConvs_A_Net import FC_Block
# Depthwise blocks keep the model compact and efficient.

# Depthwise blocks reduce parameters while keeping useful features.
class Depthwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, pool = True):
        super().__init__()

        # One filter per input channel, then a pointwise channel-mixing layer.
        self.depthwise_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding= 1)
        self.pointwise_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        # Repeat the same pattern on the output channel space.
        self.depthwise_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, groups=out_channels, padding= 1)
        self.pointwise_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, input_image):
        # Separate spatial filtering from channel mixing.
        x_1 = self.pointwise_conv1(self.depthwise_conv1(input_image))
        x_2 = self.activation(self.norm1(x_1))
        x_3 = self.pointwise_conv2(self.depthwise_conv2(x_2))
        x_4 = self.activation(self.norm2(x_3))  
        feature_map = self.pool(x_4)

        return feature_map

# Encoder_B expands channels while shrinking feature maps.
class Encoder_B(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4):
        super().__init__()
        
        # Keep a list of depthwise stages for the encoder.
        self.convblocks =  nn.ModuleList()
        current_in = in_channels
        current_out = base_channels

        # Each stage doubles the number of channels and downsamples features.
        for level in range(levels):
            is_bottom = (level == (levels -1))
            self.convblocks.append(Depthwise_conv(in_channels=current_in, out_channels=current_out, pool = not is_bottom))
            current_in = current_out
            current_out = 2* current_in

        self.out_channels = current_in
        # Summarize each channel map with a global average pool.
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)

    def forward(self, input_image):
        # Process the image through every depthwise stage.
        x = input_image
        for block in self.convblocks:
            x = block(x)

        # Reduce the 3D feature map to a single vector per sample.
        x_1 = self.gap(x)
        x_1 = x_1.flatten(1)
        return x_1

# Final model wraps encoder and classifier together.
class NodeConvs_B_Net(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4, dropout = 0.4, fc_depth = 2, num_classes = 8):
        super().__init__()

        # Create the depthwise feature encoder.
        self.encoder = Encoder_B(in_channels= in_channels, base_channels= base_channels, levels= levels)
        # Attach the shared classifier head for final class scores.
        self.Fc = FC_Block(input_units=self.encoder.out_channels, depth=fc_depth, output_units=num_classes, dropout=dropout)

    def forward(self,input_image):
        # Encode image features into a compact vector.
        feature_tensor = self.encoder(input_image)
        # Convert features into logits for each class.
        output_logits = self.Fc(feature_tensor)

        return output_logits

    def count_trainable_parameters(self):
        # Count all trainable parameters for model sizing.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Sanity check for a sample batch and parameter count.
if __name__ == "__main__":
    # Build a quick test model to confirm the architecture works.
    model = NodeConvs_B_Net(in_channels=3, base_channels=64, levels=4, num_classes=8)
    dummy_batch = torch.randn(4, 3, 64, 64)  # batch of 4 RGB 64x64 images
 
    # Verify one output vector is produced for each image.
    logits = model(dummy_batch)
    print(logits.shape)  # expected: torch.Size([4, 8])
    print(f"Total trainable parameters: {model.count_trainable_parameters():,}")