from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model - will download automatically if not present
try:
    model = YOLO("yolov8n.pt")  # Using nano version for speed
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure you have internet connection for first-time model download")
    model = None

def detect_car(image):
    """
    Detect car in image using YOLOv8
    
    Args:
        image: Input image (BGR format from cv2.imread)
        
    Returns:
        tuple: (is_car_detected, bounding_box)
               bounding_box format: (x1, y1, x2, y2) or None
    """
    if model is None:
        return False, None
    
    if image is None:
        return False, None
    
    try:
        # Run YOLO inference
        results = model(image, verbose=False)  # Turn off verbose output
        
        # Look for cars in detection results
        # COCO class 2 = car
        car_classes = [2]  # Only cars, not motorcycles, buses, or trucks
        
        best_car = None
        best_confidence = 0
        
        for r in results[0].boxes.data:
            confidence = float(r[4])  # confidence score
            cls = int(r[5])  # class id
            
            # Check if detected object is a car with good confidence
            if cls in car_classes and confidence > 0.5:
                if confidence > best_confidence:
                    best_confidence = confidence
                    x1, y1, x2, y2 = map(int, r[:4])
                    best_car = (x1, y1, x2, y2)
        
        if best_car is not None:
            return True, best_car
        else:
            return False, None
            
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return False, None

def get_multiple_cars(image):
    """
    Detect multiple cars in image (for future use)
    
    Args:
        image: Input image
        
    Returns:
        list: List of (confidence, bounding_box) tuples
    """
    if model is None:
        return []
    
    if image is None:
        return []
    
    try:
        results = model(image, verbose=False)
        cars = []
        car_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        for r in results[0].boxes.data:
            confidence = float(r[4])
            cls = int(r[5])
            
            if cls in car_classes and confidence > 0.5:
                x1, y1, x2, y2 = map(int, r[:4])
                cars.append((confidence, (x1, y1, x2, y2)))
        
        # Sort by confidence (highest first)
        cars.sort(key=lambda x: x[0], reverse=True)
        return cars
        
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return []
