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

    def forward(self, x):

        indentity = None
        skip_count = 0

        for i, (conv_layer, batchnorm_layer) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            x = conv_layer(x)

            if i in self.skip_list_C:
                identity = x

            if i in self.skip_targets_C:
                x = x + self.adjusting_layer[skip_count](identity)
                skip_count += 1

            x = batchnorm_layer(x)
            x = silu(x)
        

        x = self.output_layer(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)

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