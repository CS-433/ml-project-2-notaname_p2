import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import remove_small_objects, closing, square, diamond
from skimage.measure import label, regionprops

'''Things to add:
- first do closing with large filters to avoid fusion of objects with roads
- then apply every the removal of false positive
- remove outliers and apply closing again or the inverse
'''


def remove_false_positives(binary_mask, min_aspect_ratio=2.5):
    """
    Remove false positives such as houses, outlier pixels, or small structures.
    Now uses max/min ratios for both width/length and length/width to better handle crossing roads.
    """
    labeled_mask = label(binary_mask)
    filtered_mask = np.zeros_like(binary_mask, dtype=bool)

    for region in regionprops(labeled_mask):
        # Use major and minor axis lengths to compute aspect ratio
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        
        if minor_axis == 0:
            aspect_ratio = 1  # Prevent division by zero
        else:
            aspect_ratio = major_axis / minor_axis

        # Retain components with a high aspect ratio (long and narrow objects)
        if aspect_ratio >= min_aspect_ratio:
            filtered_mask[labeled_mask == region.label] = True

    return filtered_mask


def process_roads(raw_map, threshold=0.5, min_size=50, min_aspect_ratio=2.5, outlier_size=20, display=False):
    """
    Post-process the CNN output to straighten roads, dynamically estimate widths, and remove false positives
    and outliers *after* reconstructing roads.
    """
    # Transform input into binary
    binary_road_map = (raw_map > threshold).astype(np.uint8)

    # Remove small pixel objects
    cleaned_mask = remove_small_objects(binary_road_map > 0, min_size=min_size)

    # Fill small holes with closing (square + diamond)
    filled_map = closing(cleaned_mask, square(3))
    filled_map = closing(filled_map, diamond(3))

    # Remove false positives using aspect ratio
    filtered_map = remove_false_positives(filled_map, min_aspect_ratio=min_aspect_ratio)
    
    # Remove small outliers again after the processing
    final_map = remove_small_objects(filtered_map, min_size=outlier_size)
    
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

    return final_map