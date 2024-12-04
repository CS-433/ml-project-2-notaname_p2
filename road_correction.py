import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import remove_small_objects, closing, square, diamond
from skimage.measure import label, regionprops

'''Things to add:
- first do closing with large filters to avoid fusion of objects with roads
- then apply every the removal of false positive
- remove outliers and apply closing again or the inverse
'''


def remove_false_positives(binary_mask, min_aspect_ratio=2.5, min_area=400):
    """
    Remove false positives such as houses, outlier pixels, or small structures.
    Now uses max/min ratios for both width/length and length/width to better handle crossing roads.
    
    Args:
        binary_mask (ndarray): Binary mask of the image.
        min_aspect_ratio (float): Minimum aspect ratio (major/minor axis) to retain regions.
        min_area (int): Minimum area for a region to be retained.
        
    Returns:
        ndarray: Binary mask with false positives removed.
    """
    labeled_mask = label(binary_mask)
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)

    for region in regionprops(labeled_mask):
        # Use major and minor axis lengths to compute aspect ratio
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        
        if minor_axis == 0:
            aspect_ratio = float("inf")  # Prevent division by zero
        else:
            aspect_ratio = major_axis / minor_axis

        # Check aspect ratio and area
        if aspect_ratio >= min_aspect_ratio or region.area >= min_area:
            filtered_mask[labeled_mask == region.label] = True

    return filtered_mask



def process_roads(raw_map, threshold=0.05, outlier_size=50, shape_size=5, min_aspect_ratio=2.5, display=False):
    """
    Post-process the CNN output to straighten roads and remove false positives
    and outliers *after* reconstructing roads.
    """
    # Transform input into binary
    binary_road_map = (raw_map > threshold).astype(np.uint8)
    
    # Remove small pixel objects
    cleaned_mask = remove_small_objects(binary_road_map > 0, min_size=outlier_size)

    # Fill small holes with closing (square + diamond)
    filled_map = closing(cleaned_mask, square(shape_size))
    filled_map = closing(filled_map, diamond(shape_size))

    # Remove false positives using aspect ratio
    filtered_map = remove_false_positives(filled_map, min_aspect_ratio=min_aspect_ratio)
    
    if display:
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        axes[0].imshow(binary_road_map, cmap='gray')
        axes[0].set_title("Binary Map")
        axes[0].axis('off')
        axes[1].imshow(cleaned_mask, cmap='gray')
        axes[1].set_title("Removed Small Objects")
        axes[1].axis('off')
        axes[2].imshow(filled_map, cmap='gray')
        axes[2].set_title("Filled Small Holes")
        axes[2].axis('off')
        axes[3].imshow(filtered_map, cmap='gray')
        axes[3].set_title("After False Positive Removal")
        axes[3].axis('off')
        axes[4].imshow(final_map, cmap='gray')
        axes[4].set_title("Final Map After Outlier Removal")
        axes[4].axis('off')
        plt.tight_layout()
        plt.show()

    return filtered_map, filled_map, cleaned_mask, binary_road_map

def f1_loss_numpy(predictions, targets, epsilon=1e-7):
    """
    Compute F1 loss using NumPy.

    Args:
        predictions (np.ndarray): Predicted probabilities (float values between 0 and 1).
        targets (np.ndarray): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to binarize predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: F1 loss (1 - F1 score).
    """

    # Compute true positives, false positives, false negatives
    true_positive = np.sum(predictions * targets)
    false_positive = np.sum(predictions * (1 - targets))
    false_negative = np.sum((1 - predictions) * targets)

    # Compute precision and recall
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)

    # Compute F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    # F1 loss is 1 - F1 score
    return 1 - f1_score
