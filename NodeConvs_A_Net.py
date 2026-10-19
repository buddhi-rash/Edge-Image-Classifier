import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Compact CNN backbone for feature extraction before the final head.


# Basic 2D conv stack extracts local image features.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,kernel_size = 3, pool = True):
        super().__init__()

        # The first convolution learns basic edges and textures.
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding=1)
        # Batch normalization keeps training stable.
        self.norm1 =nn.BatchNorm2d(out_channels)
        # The second convolution increases feature depth for richer patterns.
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size= kernel_size, padding=1)
        self.norm2 =nn.BatchNorm2d(out_channels)
        # ReLU adds non-linearity after each convolution block.
        self.activation = nn.ReLU(inplace=True)
        # Pooling reduces spatial dimensions after each block.
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, input_image):
        # Apply the first convolution and activation.
        x_1 = self.activation(self.norm1(self.conv1(input_image)))
        # Apply the second convolution and activation.
        x_2 = self.activation(self.norm2(self.conv2(x_1)))
        # Downsample the feature map to keep the network compact.
        feature_map = self.pool(x_2)

        return feature_map



# Encoder reduces spatial size and increases channel depth.
class Encoder(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4):
        super().__init__()
        
        # Store a stack of convolution blocks for each stage.
        self.convblocks =  nn.ModuleList()
        current_in = in_channels
        current_out = base_channels

        # Each level doubles the feature channels and halves the map size.
        for level in range(levels):
            is_bottom = (level == (levels -1))
            self.convblocks.append(ConvBlock(in_channels=current_in, out_channels=current_out, pool = not is_bottom))
            current_in = current_out
            current_out = 2* current_in

        # Final feature channels are passed to the classifier.
        self.out_channels = current_in
        # Global average pooling converts each feature map into one value.
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)

    def forward(self, input_image):
        # Push the image through each encoder stage.
        x = input_image
        for block in self.convblocks:
            x = block(x)

        # Reduce each feature map to a single summary value.
        x_1 = self.gap(x)
        x_1 = x_1.flatten(1)
        return x_1

# Classifier head maps features to eight blood-cell classes.
class FC_Block(nn.Module):
    def __init__(self, input_units = 512, output_units = 8, depth = 2, dropout = 0.4):
        super().__init__()

        # Dropout helps prevent overfitting during training.
        layers = [nn.Dropout(dropout)]
        current_units = input_units

        # Build a small hidden network before the final output layer.
        for _ in range(depth - 1):
            hidden_units = max(input_units // 8 , output_units)
            layers += [nn.Linear(in_features=current_units, out_features= hidden_units),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout)]
            current_units = hidden_units 
        # Final linear layer produces one score per class.
        layers.append(nn.Linear(in_features= current_units, out_features= output_units))

        self.network = nn.Sequential(*layers)
        
        

    def forward(self, features):
        # Pass the encoded feature vector through the classifier.
        return self.network(features)


# Final model combines encoder and classification layers.
class NodeConvs_Net(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, levels = 4, dropout = 0.4, fc_depth = 2, num_classes = 8):
        super().__init__()

        # First create the image feature extractor.
        self.encoder = Encoder(in_channels= in_channels, base_channels= base_channels, levels= levels)
        # Then attach the final class prediction head.
        self.Fc = FC_Block(input_units=self.encoder.out_channels, depth=fc_depth, output_units=num_classes, dropout=dropout)

    def forward(self,input_image):
        # Encode image to a compact feature representation.
        feature_tensor = self.encoder(input_image)
        # Convert those features into class logits.
        output_logits = self.Fc(feature_tensor)

        return output_logits

    def count_trainable_parameters(self):
        # Sum all trainable parameters for quick model sizing.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Sanity check for a sample batch and parameter count.
if __name__ == "__main__":
    # Build a small test model to verify the forward pass.
    model = NodeConvs_Net(in_channels=3, base_channels=64, levels=4, num_classes=8)
    dummy_batch = torch.randn(4, 3, 64, 64)  # batch of 4 RGB 64x64 images
 
    # Ensure the model returns one logit vector per image.
    logits = model(dummy_batch)
    print(logits.shape)  # expected: torch.Size([4, 8])
    print(f"Total trainable parameters: {model.count_trainable_parameters():,}")

