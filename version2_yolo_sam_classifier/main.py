#!/usr/bin/env python3
"""
Version 2: YOLO + SAM + Classifier based car image validation
Uses YOLOv8 for detection, segmentation, and ML-based orientation classification
"""

import os
import sys
import json
import cv2
from datetime import datetime
from yolo_module import detect_car
from sam_module import segment_car
from orientation_classifier import classify_orientation
from utils import check_area_ratio, check_white_background, validate_image_quality

def verify_image(image_path, mode="simple"):
    """
    Comprehensive car image validation using YOLO + SAM + Classifier
    
    Args:
        image_path: Path to image
        mode: "simple" or "detailed"
        
    Returns:
        dict: Validation result
    """
    try:
        if not os.path.exists(image_path):
            return {
                "result": "Error",
                "error": "File not found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "result": "Error",
                "error": "Invalid image format",
                "timestamp": datetime.now().isoformat()
            }
        
        # Quality validation
        is_quality_ok, quality_score, quality_issues = validate_image_quality(img)
        if not is_quality_ok:
            return {
                "result": "Error",
                "error": f"Image quality issues: {', '.join(quality_issues)}",
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 1: Detect car using YOLO
        is_car, bbox = detect_car(img)
        if not is_car or bbox is None:
            result = {
                "result": "No",
                "failure_reasons": ["No car detected in image"],
                "timestamp": datetime.now().isoformat(),
                "model_used": "yolov8n + sam + classifier"
            }
            
            if mode == "detailed":
                result.update({
                    "criteria": {
                        "contains_car": False,
                        "proper_segmentation": False,
                        "white_background": False,
                        "proper_size": False,
                        "side_view": False,
                        "facing_left": False
                    },
                    "image_path": image_path,
                    "quality_score": quality_score
                })
            
            return result
        
        # Step 2: Segment car using SAM-like approach
        mask = segment_car(img, bbox)
        
        # Step 3: Check criteria
        area_ok = check_area_ratio(mask)
        background_white = check_white_background(img, mask)
        orientation_result = classify_orientation(img, bbox)
        
        side_view = orientation_result["side_view"]
        facing_left = orientation_result["facing_left"]
        
        # Collect failure reasons
        failure_reasons = []
        if not area_ok:
            failure_reasons.append("Car does not occupy sufficient portion of image (< 20%)")
        if not background_white:
            failure_reasons.append("Background is not sufficiently white")
        if not side_view:
            failure_reasons.append("Car does not appear to be in side view")
        if not facing_left:
            failure_reasons.append("Car does not appear to be facing left")
        
        # Final validation
        all_passed = all([is_car, area_ok, background_white, side_view, facing_left])
        
        result = {
            "result": "Yes" if all_passed else "No",
            "timestamp": datetime.now().isoformat(),
            "model_used": "yolov8n + sam + classifier"
        }
        
        if mode == "detailed":
            result.update({
                "criteria": {
                    "contains_car": bool(is_car),
                    "proper_segmentation": True,  # Always true if we got here
                    "white_background": bool(background_white),
                    "proper_size": bool(area_ok),
                    "side_view": bool(side_view),
                    "facing_left": bool(facing_left)
                },
                "orientation_confidence": orientation_result["confidence_scores"],
                "quality_score": quality_score,
                "image_path": image_path
            })
            
            if not all_passed:
                result["failure_reasons"] = failure_reasons
        elif not all_passed:
            result["failure_reasons"] = failure_reasons
        
        return result
        
    except Exception as e:
        return {
            "result": "Error",
            "error": f"Processing error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def validate_multiple_images(directory_path, mode="simple"):
    """
    Validate multiple images in a directory
    
    Args:
        directory_path: Path to directory containing images
        mode: "simple" or "detailed"
        
    Returns:
        dict: Results for all images
    """
    if not os.path.exists(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    results = {}
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    image_files = [f for f in os.listdir(directory_path) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        return {"error": "No image files found in directory"}
    
    print(f"Found {len(image_files)} images to validate...")
    print(f"Mode: {mode.upper()}")
    print()
    
    for filename in sorted(image_files):
        file_path = os.path.join(directory_path, filename)
        print(f"Validating {filename}...")
        
        result = verify_image(file_path, mode)
        
        # Store result with clean key
        clean_key = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        
        if mode == "simple":
            results[clean_key] = {"result": result["result"]}
        else:
            results[filename] = result
        
        # Print progress
        status = result["result"]
        if status == "Yes":
            print(f"  → {status}")
        elif status == "No":
            reasons = result.get("failure_reasons", [])
            print(f"  → {status}")
            for reason in reasons[:2]:  # Show first 2 reasons
                print(f"    • {reason}")
        else:  # Error
            print(f"  → Error: {result.get('error', 'Unknown error')}")
    
    return results

def main():
    print("🚗 Car Image Validation - Version 2 (YOLO + SAM + Classifier)")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        # Interactive mode
        print("Choose validation mode:")
        print("1. Simple mode (result: Yes/No only)")
        print("2. Detailed mode (full criteria breakdown)")
        print()
        
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            mode = "simple" if choice == "1" else "detailed"
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        
        # Default to test images
        directory_path = "../assets/test_images/"
        print(f"\nNo path specified, using default: {directory_path}")
    else:
        # Command line mode
        path = sys.argv[1]
        mode = "detailed"  # Default to detailed for single image
        
        if os.path.isfile(path):
            # Single image
            result = verify_image(path, mode)
            print(json.dumps(result, indent=2))
            return
        else:
            directory_path = path
    
    print("=" * 60)
    print(f"Validating all images in directory: {directory_path}")
    
    # Validate multiple images
    results = validate_multiple_images(directory_path, mode)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_results_{mode}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if 
                 (isinstance(r, dict) and r.get("result") == "Yes") or 
                 (isinstance(r, dict) and r.get("result") == "Yes"))
    failed = sum(1 for r in results.values() if 
                 (isinstance(r, dict) and r.get("result") == "No") or 
                 (isinstance(r, dict) and r.get("result") == "No"))
    errors = total - passed - failed
    
    print(f"\nSummary:")
    print(f"  Total images: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")
    
    if failed > 0 and mode == "detailed":
        print(f"\n💡 Check detailed failure reasons in {filename}")

if __name__ == "__main__":
    main()
