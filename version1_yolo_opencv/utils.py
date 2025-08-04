import cv2
import numpy as np

def check_size_ratio(bbox, image_shape):
    """
    Check if car occupies at least 1/4 of the image area
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_shape: Image shape (height, width, channels)
        
    Returns:
        bool: True if car is large enough
    """
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    img_area = image_shape[0] * image_shape[1]  # height * width
    ratio = box_area / img_area
    # Adjusted to 25% threshold for reasonable car sizes
    return ratio >= 0.25

def check_background_white(image, bbox):
    """
    Check if background (non-car area) is predominantly white
    
    Args:
        image: Input image
        bbox: Car bounding box to exclude from background analysis
        
    Returns:
        bool: True if background is sufficiently white
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Create mask excluding the car area
    mask = np.ones((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 0
    
    # Get background pixels
    bg_pixels = image[mask == 1]
    
    if len(bg_pixels) == 0:
        return False
    
    # Check if background is white-ish (allowing for some shadows/lighting)
    if len(image.shape) == 3:  # Color image
        avg_color = np.mean(bg_pixels, axis=0)
        # Relaxed threshold for realistic lighting conditions
        return np.all(avg_color > 180)
    else:  # Grayscale
        avg_brightness = np.mean(bg_pixels)
        return avg_brightness > 180

def heuristic_orientation_check(bbox, image=None):
    """
    Check if car appears to be in side view based on aspect ratio
    
    Args:
        bbox: Car bounding box (x1, y1, x2, y2)
        image: Optional image (unused, kept for compatibility)
        
    Returns:
        bool: True if car appears to be in side view
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # For side view, width should be significantly larger than height
    aspect_ratio = width / height if height > 0 else 0
    
    # Cars in side view typically have aspect ratio >= 1.8
    return aspect_ratio >= 1.8

def check_car_orientation_left(bbox, image_shape):
    """
    Simple heuristic to check if car appears to be facing left
    
    Args:
        bbox: Car bounding box (x1, y1, x2, y2)
        image_shape: Image shape (height, width, channels)
        
    Returns:
        bool: True if car appears to be facing left
    """
    x1, y1, x2, y2 = bbox
    car_center_x = (x1 + x2) // 2
    image_width = image_shape[1]
    
    # Simple position-based heuristic: 
    # Cars facing left are typically positioned in center-right area
    relative_position = car_center_x / image_width
    
    # Car should not be too close to edges and should have reasonable positioning
    return 0.3 <= relative_position <= 0.8

def validate_image_quality(image):
    """
    Basic image quality checks
    
    Args:
        image: Input image
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if image is None:
        return False, "Could not load image"
    
    h, w = image.shape[:2]
    
    # Check minimum resolution
    if h < 100 or w < 100:
        return False, "Image resolution too low"
    
    # Check if image is too dark or too bright
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30:
        return False, "Image too dark"
    elif mean_brightness > 250:
        return False, "Image too bright"
    
    return True, "OK"
