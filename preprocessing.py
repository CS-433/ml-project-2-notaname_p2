
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