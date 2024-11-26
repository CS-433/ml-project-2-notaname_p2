import numpy as np
import pandas as pd

import torch

from torch.nn.functional import relu

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

    def __init__(self, in_channels, list_conv_param, skip_list=None, skip_targets = None):
        super(Conv_test, self).__init__()
        self.in_channels = in_channels

        self.conv_list = torch.nn.ModuleList()
        self.batchnorm_list = torch.nn.ModuleList()
        self.skip_list = skip_list if skip_list is not None else []
        self.skip_targets = skip_targets if skip_targets is not None else []

        assert len(self.skip_list) == len(self.skip_targets)

        self.skip_proj = torch.nn.ModuleList()

        if self.skip_list is not None:
            self.skip_connect = torch.nn.ModuleList()

        for i, conv_params in enumerate(list_conv_param):
    
            conv_layer = torch.nn.Conv2d(in_channels=in_channels, **conv_params)
            self.conv_list.append(conv_layer)

            out_channels = conv_params["out_channels"]
            batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
            self.batchnorm_list.append(batchnorm_layer)
            
            if i in self.skip_targets:
                self.skip_proj.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))

            in_channels = out_channels

        self.output_layer = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1
        )

    def forward(self, x):

        skip_outputs = {}

        for i, (conv_layer, batchnorm_layer) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            x = conv_layer(x)
            x = batchnorm_layer(x)
            x = relu(x)

            if i in self.skip_list:
                skip_outputs[i] = x

            if i in self.skip_targets:
                skip_index = self.skip_targets.index(i)
                skip_source = self.skip_list[skip_index]

                skip_out = skip_outputs[skip_source]
                if skip_out.shape != x.shape:
                    skip_out = self.skip_proj[skip_index](skip_out)

                x = x + skip_out

        x = self.output_layer(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)

        return x

class Road_data(Dataset):
    def __init__(self, images, labels):
        # necessary to convert numpy style images to tensor style images 
        images = [torch.tensor(image).permute(2, 0, 1) for image in images]
        labels = [torch.tensor(label) for label in labels]
        
        self.images = images  
        self.labels = labels  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]