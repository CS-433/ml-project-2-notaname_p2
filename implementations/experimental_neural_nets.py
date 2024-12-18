"""
This module implements various components for building and training convolutional models,
including a custom convolutional model (Conv_test), U-Net architecture, and a dataset class (Road_data).

Contents:
1. Conv_test: A convolutional model with configurable skip connections and BatchNorm layers.
2. DoubleConv, Down, Up, OutConv: Modular components for building a U-Net-like architecture.
3. UNet: A classical U-Net model implementation for segmentation tasks.
4. Road_data: A PyTorch dataset for loading and processing road segmentation images and labels.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import relu, tanh, leaky_relu, silu
from torch.utils.data import Dataset


def set_seed():
    """
    Sets the random seed for reproducibility in PyTorch and NumPy.

    Ensures that experiments yield consistent results by fixing the seed for random number generators.
    """
    torch.manual_seed(0)
    np.random.seed(0)


class Conv_test(torch.nn.Module):
    """
    A convolutional model with configurable Conv2D and BatchNorm2D layers, supporting skip connections.

    Parameters:
    ----------
    in_channels : int
        Number of input channels for the first convolutional layer.
    list_conv_param : list of dict
        List of dictionaries specifying parameters for each Conv2D layer.
    skip_list_C : list of int, optional
        Indices of layers where skip connections originate.
    skip_targets_C : list of int, optional
        Indices of layers where skip connections terminate.

    Example:
    --------
    model = Conv_test(
        in_channels=3,
        list_conv_param=[{'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}],
        skip_list_C=[0],
        skip_targets_C=[1]
    )
    """
    def __init__(self, in_channels, list_conv_param, skip_list_C=None, skip_targets_C=None):
        super(Conv_test, self).__init__()

        self.conv_list = torch.nn.ModuleList()
        self.batchnorm_list = torch.nn.ModuleList()

        self.skip_list_C = skip_list_C if skip_list_C is not None else []
        self.skip_targets_C = skip_targets_C if skip_targets_C is not None else []
        skip_insize_C = []  # To store intermediate channel sizes for skip connections
        self.adjusting_layer = torch.nn.ModuleList()

        # Ensure the number of skip sources matches the number of skip targets
        assert len(self.skip_list_C) == len(self.skip_targets_C)

        skip_count = 0
        for i, conv_params in enumerate(list_conv_param):

            conv_layer = torch.nn.Conv2d(in_channels=in_channels, **conv_params)
            self.conv_list.append(conv_layer)

            out_channels = conv_params["out_channels"]
            if i in self.skip_list_C:
                skip_insize_C.append(out_channels)

            if i in self.skip_targets_C:
                self.adjusting_layer.append(torch.nn.Conv2d(
                    in_channels=skip_insize_C[skip_count],
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1
                ))
                skip_count += 1

            batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
            self.batchnorm_list.append(batchnorm_layer)

            in_channels = out_channels

        self.output_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the Conv_test model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        -------
        torch.Tensor
            Output tensor after applying the model.
        """
        identity = None
        skip_count = 0

        for i, (conv_layer, batchnorm_layer) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            x = conv_layer(x)

            # Handle skip connections
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
    Double convolution block: Conv -> BatchNorm -> SiLU -> Conv -> BatchNorm -> SiLU

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
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
    Downsampling block: MaxPool -> DoubleConv

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
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
    Upsampling block: Upsample -> Concatenate -> DoubleConv

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle possible mismatches in spatial dimensions due to rounding errors
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    """
    Output block: Final convolution layer with optional activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (typically 1 for binary segmentation).
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
    U-Net model implementation for segmentation tasks.

    Parameters:
    ----------
    n_channels : int, default=3
        Number of input channels (e.g., 3 for RGB images).
    base_c : int, default=64
        Number of filters in the first layer.
    num_layers : int, default=4
        Number of layers in the encoder and decoder.

    Example:
    --------
    model = UNet(n_channels=3, base_c=64, num_layers=4)
    """
    def __init__(self, n_channels=3, base_c=64, num_layers=4):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.num_layers = num_layers

        self.inc = DoubleConv(n_channels, base_c)
        self.downs = torch.nn.ModuleList()
        in_channels = base_c
        for _ in range(num_layers):
            out_channels = in_channels * 2
            self.downs.append(Down(in_channels, out_channels))
            in_channels = out_channels

        self.ups = torch.nn.ModuleList()
        for _ in range(num_layers):
            out_channels = in_channels // 2
            self.ups.append(Up(in_channels, out_channels))
            in_channels = out_channels

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
    """
    PyTorch Dataset for road segmentation tasks.

    Parameters:
    ----------
    images : list of np.ndarray
        List of input images (H x W x C).
    labels : list of np.ndarray
        List of segmentation masks (H x W).

    Note:
    -----
    The images and labels are converted into PyTorch tensors and transposed to
    channel-first format for compatibility with PyTorch models.
    """
    def __init__(self, images, labels):
        # Convert images and labels to tensors with channel-first format
        images = [torch.tensor(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous() for image in images]
        labels = [torch.tensor(np.ascontiguousarray(label)).contiguous() for label in labels]

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]