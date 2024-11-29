import numpy as np

from skimage.morphology import remove_small_objects, closing, square, diamond
from skimage.measure import label, regionprops


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


def process_roads(raw_map, threshold=0.5, min_size=50, min_aspect_ratio=2.5, outlier_size=20):
    """
    Post-process the CNN output to straighten roads, dynamically estimate widths, and remove false positives
    and outliers *after* reconstructing roads.
    """
    # Transform input into binary
    binary_road_map = (raw_map > threshold).astype(np.uint8)

    # Remove small pixel objects
    cleaned_mask = remove_small_objects(binary_road_map > 0, min_size=min_size)

    # Fill small holes with closing function (square for horizontal roads, diamonds for diagonals)
    filled_map = closing(cleaned_mask, square(3))
    filled_map = closing(filled_map, diamond(3))

    # Remove false positives (houses and small rectangles)
    filtered_map = remove_false_positives(filled_map, min_aspect_ratio=min_aspect_ratio)

    # Removal of outliers left or caused by remove_false_positives
    final_map = remove_small_objects(filtered_map, min_size=outlier_size)

    return final_map