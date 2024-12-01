import numpy as np
import pandas as pd

import torch

from torch.nn.functional import relu, tanh, leaky_relu, silu

from torch.utils.data import Dataset

class Conv_test(torch.nn.Module):
    """
    A convolutional model with multiple Conv2D and BatchNorm2D layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels for the first convolutional layer.

    list_conv_param : list of dict
        List of dictionaries where each dictionary specifies the parameters for a Conv2D layer
        (e.g., {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}).
    """

    def __init__(self, in_channels, list_conv_param, skip_list_C=None, skip_targets_C = None):
        super(Conv_test, self).__init__()

        self.conv_list = torch.nn.ModuleList()
        self.batchnorm_list = torch.nn.ModuleList()

        self.skip_list_C = skip_list_C if skip_list_C is not None else []
        self.skip_targets_C = skip_targets_C if skip_targets_C is not None else []
        skip_insize_C = []
        self.adjusting_layer = torch.nn.ModuleList()

        assert len(self.skip_list_C) == len(self.skip_targets_C)

        skip_count = 0

        for i, conv_params in enumerate(list_conv_param):
    
            conv_layer = torch.nn.Conv2d(in_channels=in_channels, **conv_params)
            self.conv_list.append(conv_layer)

            out_channels = conv_params["out_channels"]

            if i in self.skip_list_C:
                skip_insize_C.append(out_channels)

            if i in self.skip_targets_C:
                self.adjusting_layer.append(torch.nn.Conv2d(in_channels=skip_insize_C[skip_count], out_channels=out_channels, kernel_size=1, stride=1))
                skip_count += 1

            batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
            self.batchnorm_list.append(batchnorm_layer)

            in_channels = out_channels

        self.output_layer = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1
        )

        #self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights non-randomly."""
        for layer in self.conv_list:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.adjusting_layer:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.constant_(layer.weight, 0.2)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)

        if isinstance(self.output_layer, torch.nn.Conv2d):
            torch.nn.init.constant_(self.output_layer.weight, 0.3) 
            if self.output_layer.bias is not None:
                torch.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):

        identity = None
        skip_count = 0

        for i, (conv_layer, batchnorm_layer) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            x = conv_layer(x)

            if i in self.skip_list_C:
                identity = x

            if i in self.skip_targets_C:
                x = x + self.adjusting_layer[skip_count](identity)
                skip_count += 1

            x = batchnorm_layer(x)
            x = relu(x)
        

        x = self.output_layer(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)

        return x


class DoubleConv(torch.nn.Module):
    """
    Double convolution: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(torch.nn.Module):
    """
    Downsampling: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)

class Up(torch.nn.Module):
    """
    Upsampling: Transposed Conv -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # In case of dimension missmatch
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])

                                
        x1 = self.conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    """
    Output Convolution: Conv -> Activation (optional)
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)
        return x

class UNet(torch.nn.Module):
    """
    U-Net Model with flexible number of layers
    """
    def __init__(self, n_channels=3, base_c=64, num_layers=4):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.num_layers = num_layers

        # Encoder
        self.inc = DoubleConv(n_channels, base_c)
        self.downs = torch.nn.ModuleList()  # To store the down-sampling layers
        in_channels = base_c
        print('down')
        for i in range(num_layers):
            out_channels = in_channels * 2
            self.downs.append(Down(in_channels, out_channels))
            print((in_channels,out_channels))
            in_channels = out_channels

        # Decoder
        self.ups = torch.nn.ModuleList()  # To store the up-sampling layers
        print('Up')
        for i in range(num_layers):
            out_channels = in_channels // 2
            self.ups.append(Up(in_channels, out_channels))
            print((in_channels,out_channels))
            in_channels = out_channels

        # Output
        self.outc = OutConv(base_c, 1)

    def forward(self, x):
        x1 = self.inc(x)
        encodings = [x1]
        for down in self.downs:
            encodings.append(down(encodings[-1]))

        x = encodings[-1]
        for i, up in enumerate(self.ups):
            x = up(x, encodings[-i-2])

        x = self.outc(x)
        return x


class Road_data(Dataset):
    def __init__(self, images, labels):
        # necessary to convert numpy style images to tensor style images 
        images = [torch.tensor(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous() for image in images]
        labels = [torch.tensor(np.ascontiguousarray(label)).contiguous() for label in labels]
        
        self.images = images  
        self.labels = labels  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]