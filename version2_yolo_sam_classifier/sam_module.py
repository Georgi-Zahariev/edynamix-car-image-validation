import cv2
import numpy as np

def segment_car(image, bbox):
    """
    Car segmentation using GrabCut-based approach
    
    Args:
        image: Input image
        bbox: Car bounding box from YOLO
        
    Returns:
        numpy.ndarray: Binary mask where 1=car, 0=background
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Initialize mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract car region
    car_region = image[y1:y2, x1:x2]
    
    try:
        # Use bounding box with minimal refinement
        # Exclude clearly white background pixels
        white_threshold = 240  # Higher threshold - only exclude very white pixels
        
        # Check all three channels - pixel must be very white in ALL channels
        very_white = np.all(car_region >= white_threshold, axis=2)
        
        # Create mask: include everything except very white pixels
        car_mask = (~very_white).astype(np.uint8)
        
        # Ensure we have enough car pixels (at least 60% of bounding box)
        car_pixels = np.sum(car_mask > 0)
        total_pixels = car_mask.size
        
        if car_pixels < total_pixels * 0.6:
            # If excluding white pixels removes too much, use full bounding box
            # This ensures we don't lose valid car pixels due to over-segmentation
            car_mask = np.ones_like(car_mask, dtype=np.uint8)
            print(f"    Debug: Using full bounding box (white exclusion removed too much)")
        
        # Place the segmented car back into the full image mask
        mask[y1:y2, x1:x2] = car_mask
        
    except Exception as e:
        # Fallback to simple bounding box if anything fails
        print(f"    Debug: Segmentation failed, using bounding box: {e}")
        mask[y1:y2, x1:x2] = 1
    
    return mask
