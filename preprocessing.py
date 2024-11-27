
import numpy as np


def rotate_train_data(imgs, gt_imgs):
    # Rotate the data by the given angle
    augmented_imgs = []
    augmented_gt_imgs = []

    for image,y in zip(imgs, gt_imgs):
        # Original image (0 degrees)
        augmented_imgs.append(image)
        augmented_gt_imgs.append(y)
        # Rotate 90 degrees
        rotated_90 = np.rot90(image, k=3)  # Clockwise
        augmented_imgs.append(rotated_90)
        augmented_gt_imgs.append(np.rot90(y, k=3))

        # Rotate 180 degrees
        rotated_180 = np.rot90(image, k=2)  # 180 degrees
        augmented_imgs.append(rotated_180)
        augmented_gt_imgs.append(np.rot90(y, k=2))
        # Rotate 270 degrees
        rotated_270 = np.rot90(image, k=1)  # Counterclockwise
        augmented_imgs.append(rotated_270)
        augmented_gt_imgs.append(np.rot90(y, k=1))

    print ("Number of images: ", len(augmented_imgs))
    print ("Number of groundtruth images: ", len(augmented_gt_imgs))
    return augmented_imgs, augmented_gt_imgs


def flip_train_data(imgs, gt_imgs):
    augmented_imgs = []
    augmented_gt_imgs = []

    for image, y in zip(imgs, gt_imgs):
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
    augmented_imgs = []
    augmented_gt_imgs = []

    for image, y in zip(imgs, gt_imgs):
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

    return augmented_imgs, augmented_gt_imgs

