"""
Data Augmentation for Image Segmentation

This script contains functions for augmenting training data used in 
image segmentation tasks. The augmentation techniques include:

1. Rotating images and their corresponding ground truth labels.
2. Flipping images horizontally, vertically, and both.
3. Adjusting brightness levels of images.

Each function ensures the output data remains consistent with the input 
in terms of structure and ground truth alignment.

Functions:
----------
- rotate_train_data: Augments data by rotating images and labels.
- flip_train_data: Augments data by flipping images and labels.
- adjust_brightness_contrast: Augments data by modifying image brightness.
"""

import numpy as np


def rotate_train_data(imgs, gt_imgs):
    """
    Augments the dataset by rotating images and ground truth labels.

    Rotates each image and its corresponding ground truth by 90, 180, 
    and 270 degrees (clockwise and counterclockwise).

    Parameters:
    ----------
    imgs : list of np.ndarray
        List of input images (H x W x C).
    gt_imgs : list of np.ndarray
        List of corresponding ground truth images (H x W).

    Returns:
    -------
    tuple
        A tuple containing two lists:
        - augmented_imgs: List of augmented images.
        - augmented_gt_imgs: List of augmented ground truth labels.
    """
    augmented_imgs = []
    augmented_gt_imgs = []

    for image, y in zip(imgs, gt_imgs):
        # Original image (0 degrees)
        augmented_imgs.append(image)
        augmented_gt_imgs.append(y)

        # Rotate 90 degrees clockwise
        rotated_90 = np.rot90(image, k=3)
        augmented_imgs.append(rotated_90)
        augmented_gt_imgs.append(np.rot90(y, k=3))

        # Rotate 180 degrees
        rotated_180 = np.rot90(image, k=2)
        augmented_imgs.append(rotated_180)
        augmented_gt_imgs.append(np.rot90(y, k=2))

        # Rotate 270 degrees clockwise (90 degrees counterclockwise)
        rotated_270 = np.rot90(image, k=1)
        augmented_imgs.append(rotated_270)
        augmented_gt_imgs.append(np.rot90(y, k=1))

    print("Number of images: ", len(augmented_imgs))
    print("Number of groundtruth images: ", len(augmented_gt_imgs))
    return augmented_imgs, augmented_gt_imgs


def flip_train_data(imgs, gt_imgs):
    """
    Augments the dataset by flipping images and ground truth labels.

    Applies horizontal, vertical, and combined flips to each image and
    its corresponding ground truth label.

    Parameters:
    ----------
    imgs : list of np.ndarray
        List of input images (H x W x C).
    gt_imgs : list of np.ndarray
        List of corresponding ground truth images (H x W).

    Returns:
    -------
    tuple
        A tuple containing two lists:
        - augmented_imgs: List of augmented images.
        - augmented_gt_imgs: List of augmented ground truth labels.
    """
    augmented_imgs = []
    augmented_gt_imgs = []

    for image, y in zip(imgs, gt_imgs):
        # Original image
        augmented_imgs.append(np.ascontiguousarray(image))
        augmented_gt_imgs.append(np.ascontiguousarray(y))

        # Horizontal flip
        flipped_h = np.ascontiguousarray(np.fliplr(image))
        augmented_imgs.append(flipped_h)
        augmented_gt_imgs.append(np.ascontiguousarray(np.fliplr(y)))

        # Vertical flip
        flipped_v = np.ascontiguousarray(np.flipud(image))
        augmented_imgs.append(flipped_v)
        augmented_gt_imgs.append(np.ascontiguousarray(np.flipud(y)))

        # Horizontal and vertical flip
        flipped_hv = np.ascontiguousarray(np.flipud(np.fliplr(image)))
        augmented_imgs.append(flipped_hv)
        augmented_gt_imgs.append(np.ascontiguousarray(np.flipud(np.fliplr(y))))

    print("Number of images after flipping: ", len(augmented_imgs))
    print("Number of groundtruth images after flipping: ", len(augmented_gt_imgs))
    return augmented_imgs, augmented_gt_imgs


def adjust_brightness_contrast(imgs, gt_imgs, factor=0.2):
    """
    Augments the dataset by adjusting the brightness of images.

    Increases and decreases the brightness of each image by the given factor,
    clipping the pixel values to stay within the valid range [0, 1].

    Parameters:
    ----------
    imgs : list of np.ndarray
        List of input images (H x W x C).
    gt_imgs : list of np.ndarray
        List of corresponding ground truth images (H x W).
    factor : float, default=0.2
        Brightness adjustment factor.

    Returns:
    -------
    tuple
        A tuple containing two lists:
        - augmented_imgs: List of augmented images with adjusted brightness.
        - augmented_gt_imgs: List of ground truth labels (unchanged).
    """
    augmented_imgs = []
    augmented_gt_imgs = []

    for image, y in zip(imgs, gt_imgs):
        # Original image
        augmented_imgs.append(np.ascontiguousarray(image))
        augmented_gt_imgs.append(np.ascontiguousarray(y))

        # Increase brightness
        brighter_image = np.clip(image + factor, 0, 1)
        augmented_imgs.append(np.ascontiguousarray(brighter_image))
        augmented_gt_imgs.append(np.ascontiguousarray(y))

        # Decrease brightness
        darker_image = np.clip(image - factor, 0, 1)
        augmented_imgs.append(np.ascontiguousarray(darker_image))
        augmented_gt_imgs.append(np.ascontiguousarray(y))

    print("Number of images after brightness adjustment: ", len(augmented_imgs))
    print("Number of groundtruth images after brightness adjustment: ", len(augmented_gt_imgs))
    return augmented_imgs, augmented_gt_imgs