import cv2
import numpy as np

def check_area_ratio(mask):
    """
    Check if car occupies at least 20% of the image area
    
    Args:
        mask: Binary mask where 1=car, 0=background
        
    Returns:
        bool: True if car area is sufficient
    """
    total_pixels = mask.size
    car_pixels = np.sum(mask == 1)
    ratio = car_pixels / total_pixels
    
    # Car must occupy at least 20% of image area
    return ratio >= 0.20

def check_white_background(image, mask):
    """
    Check if background is predominantly white
    
    Args:
        image: Original image
        mask: Binary mask where 1=car, 0=background
        
    Returns:
        bool: True if background is sufficiently white
    """
    # Get background pixels (where mask == 0)
    bg_pixels = image[mask == 0]
    
    if len(bg_pixels) == 0:
        return False
    
    # Calculate average color of background
    avg_color = np.mean(bg_pixels, axis=0)
    
    # More sophisticated white detection
    # Check if all RGB channels are above threshold and relatively balanced
    white_threshold = 200
    color_balance_threshold = 20  # Max difference between channels
    
    is_bright = np.all(avg_color > white_threshold)
    is_balanced = np.max(avg_color) - np.min(avg_color) < color_balance_threshold
    
    return is_bright and is_balanced

def validate_image_quality(image):
    """
    Image quality validation
    
    Args:
        image: Input image
        
    Returns:
        tuple: (is_valid, quality_score, issues)
    """
    if image is None:
        return False, 0.0, ["Could not load image"]
    
    h, w = image.shape[:2]
    issues = []
    quality_score = 1.0
    
    # Resolution check
    if h < 200 or w < 200:
        issues.append("Low resolution")
        quality_score -= 0.3
    
    # Brightness analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 50:
        issues.append("Too dark")
        quality_score -= 0.2
    elif mean_brightness > 240:
        issues.append("Too bright/overexposed")
        quality_score -= 0.2
    
    # Contrast check
    contrast = np.std(gray)
    if contrast < 20:
        issues.append("Low contrast")
        quality_score -= 0.2
    
    # Blur detection using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        issues.append("Image appears blurry")
        quality_score -= 0.3
    
    quality_score = max(0.0, quality_score)
    is_valid = quality_score > 0.5 and len(issues) < 3
    
    return is_valid, quality_score, issues
