import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image

import pandas as pd

import torch



def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def getting_datasets ():
    root_dir = "data/training/"

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n =  len(files)  # Load maximum 100 images
    print("Loading X_train " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
   
    gt_dir = root_dir + "groundtruth/"
    print("Loading Y_train" + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
   
    # Getting the data to test

    # Define the root directory containing the test image folders
    test_dir = "data/test_set_images/"

    # List all subdirectories (each containing one image)
    test_folders = [folder for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))]

    # Calculate the number of test images (based on the number of folders)
    n_test = len(test_folders)
    print("Loading X_test" + str(n_test) + " images")

    # Iterate over each folder, find the image, and load it
    test_imgs = []

    for folder in test_folders:
        folder_path = os.path.join(test_dir, folder)
    
        # List all files in the folder (assuming only one image per folder)
        image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    
        # Assuming there's only one image per folder, get the image file path
        if image_files:
            image_path = os.path.join(folder_path, image_files[0])
            test_imgs.append(load_image(image_path))
    return imgs, gt_imgs, test_imgs
