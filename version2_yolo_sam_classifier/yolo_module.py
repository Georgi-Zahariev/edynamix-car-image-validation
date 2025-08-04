from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_car(image):
    """
    Detect car in image using YOLOv8
    
    Args:
        image: Input image
        
    Returns:
        tuple: (is_car_detected, best_bbox_with_confidence)
    """
    results = model(image, verbose=False)
    
    if len(results[0].boxes) == 0:
        return False, None
    
    car_detections = []
    
    # Find all car detections (class 2 in COCO)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls == 2 and conf > 0.5:  # Car class with confidence > 50%
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })
    
    if not car_detections:
        return False, None
    
    # Return the highest confidence detection
    best_detection = max(car_detections, key=lambda x: x['confidence'])
    return True, best_detection['bbox']
