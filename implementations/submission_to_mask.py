#!/usr/bin/python
import os
import sys
from PIL import Image
import math
import matplotlib.image as mpimg
import numpy as np

# File containing labels and predictions
label_file = 'dummy_submission.csv'

# Image dimensions and parameters
h = 16  # Block height
w = h   # Block width
imgwidth = int(math.ceil((600.0 / w)) * w)  # Image width (rounded to nearest multiple of w)
imgheight = int(math.ceil((600.0 / h)) * h)  # Image height (rounded to nearest multiple of h)
nc = 3  # Number of channels (not used in the current implementation)


def binary_to_uint8(img):
    """
    Converts a binary array to uint8 format.
    """
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(image_id):
    """
    Reconstructs an image from label predictions stored in a CSV file.

    Reads the label file, extracts predictions for the given image ID, and 
    reconstructs the corresponding binary image. The reconstructed image is 
    saved as a PNG file.

    Parameters:
    ----------
    image_id : int
        Identifier of the image to reconstruct.

    Returns:
    -------
    np.ndarray
        Reconstructed binary image as a NumPy array.
    """

    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    image_id_str = f'{image_id:03d}_'

    for line in lines[1:]:  
        if image_id_str not in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)

        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    output_filename = f'prediction_{image_id:03d}.png'
    Image.fromarray(im).save(output_filename)

    return im

   
