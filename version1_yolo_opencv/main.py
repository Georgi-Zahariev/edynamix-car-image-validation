#!/usr/bin/env python3
"""
Version 1: YOLO + OpenCV based car image validation

Simple baseline implementation using YOLOv8 for car detection 
and OpenCV for basic image analysis. See README.md for limitations.
"""

import os
import sys
import json
import cv2
from datetime import datetime
from yolov8_loader import detect_car
from utils import check_background_white, check_size_ratio, heuristic_orientation_check, check_car_orientation_left, validate_image_quality

def get_user_choice():
    """Ask user for validation mode"""
    print("🚗 Car Image Validation - Version 1 (YOLO + OpenCV)")
    print("=" * 50)
    print("Choose validation mode:")
    print("1. Simple mode (result: Yes/No only)")
    print("2. Detailed mode (full criteria breakdown)")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "simple"
        elif choice == "2":
            return "detailed"
        else:
            print("Please enter 1 or 2")

def verify_image(image_path: str, mode: str = "detailed") -> dict:
    """
    Verify if a car image meets validation criteria using YOLO + OpenCV
    
    Args:
        image_path: Path to the image file
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

        # Validate image quality
        is_valid, quality_msg = validate_image_quality(img)
        if not is_valid:
            return {
                "result": "Error",
                "error": f"Image quality issue: {quality_msg}",
                "timestamp": datetime.now().isoformat()
            }

        # Step 1: Detect car using YOLO
        is_car, bbox = detect_car(img)
        if not is_car or bbox is None:
            result = {
                "result": "No",
                "criteria": {
                    "contains_car": False,
                    "side_view": False,
                    "white_background": False,
                    "proper_size": False,
                    "correct_orientation": False
                },
                "failure_reasons": ["No car detected in image"],
                "timestamp": datetime.now().isoformat(),
                "model_used": "yolov8n + opencv",
                "image_path": image_path
            }
            return result

        # Step 2: Check other criteria
        size_ok = check_size_ratio(bbox, img.shape)
        background_white = check_background_white(img, bbox)
        side_view = heuristic_orientation_check(bbox, img)
        facing_left = check_car_orientation_left(bbox, img.shape)

        # Collect failure reasons
        failure_reasons = []
        if not size_ok:
            failure_reasons.append("Car does not occupy sufficient portion of image (< 1/4)")
        if not background_white:
            failure_reasons.append("Background is not sufficiently white")
        if not side_view:
            failure_reasons.append("Car does not appear to be in side view (width/height ratio)")
        if not facing_left:
            failure_reasons.append("Car does not appear to be facing left")

        # Final result
        all_passed = all([is_car, size_ok, background_white, side_view, facing_left])
        
        result = {
            "result": "Yes" if all_passed else "No",
            "criteria": {
                "contains_car": bool(is_car),
                "side_view": bool(side_view),
                "white_background": bool(background_white),
                "proper_size": bool(size_ok),
                "correct_orientation": bool(facing_left)
            },
            "timestamp": datetime.now().isoformat(),
            "model_used": "yolov8n + opencv",
            "image_path": image_path
        }
        
        if not all_passed:
            result["failure_reasons"] = failure_reasons
            
        return result

    except Exception as e:
        return {
            "result": "Error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path
        }

def validate_multiple_images(image_dir: str, mode: str = "detailed") -> dict:
    """
    Validate all images in a directory
    
    Args:
        image_dir: Directory containing images
        mode: "simple" or "detailed"
        
    Returns:
        dict: Results for all images
    """
    results = {}
    
    if not os.path.exists(image_dir):
        return {"error": f"Directory not found: {image_dir}"}
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in os.listdir(image_dir) 
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to validate...")
    print(f"Mode: {mode.upper()}")
    print()
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Validating {image_file}...")
        
        result = verify_image(image_path, mode)
        results[image_file] = result
        
        # Print immediate result
        status = result.get("result", "Error")
        print(f"  → {status}")
        
        # Show failure reasons only in detailed mode console output
        if mode == "detailed" and status == "No" and "failure_reasons" in result:
            for reason in result["failure_reasons"]:
                print(f"    • {reason}")
        elif status == "Error" and "error" in result:
            print(f"    • Error: {result['error']}")
    
    return results

def main():
    """Main function to run car image validation"""
    
    # Get validation mode from user
    mode = get_user_choice()
    
    # Default to test images directory if no argument provided
    if len(sys.argv) < 2:
        input_path = "../assets/test_images/"
        print(f"\nNo path specified, using default: {input_path}")
    else:
        input_path = sys.argv[1]
        print(f"\nProcessing: {input_path}")
    
    print("=" * 50)
    
    if os.path.isfile(input_path):
        # Single image validation
        print(f"Validating single image: {input_path}")
        result = verify_image(input_path, mode)
        
        # Print result with details based on mode
        if mode == "simple":
            # Simple output: just the result
            print(f'"{result["result"]}"')
        else:
            # Detailed output: full JSON
            print(json.dumps(result, indent=2))
        
        # Show failure reasons only in detailed mode
        if mode == "detailed" and isinstance(result, dict) and result.get("result") == "No" and "failure_reasons" in result:
            print("\nFailure reasons:")
            for reason in result["failure_reasons"]:
                print(f"  • {reason}")
        
    elif os.path.isdir(input_path):
        # Multiple image validation
        print(f"Validating all images in directory: {input_path}")
        results = validate_multiple_images(input_path, mode)
        
        # Save results to file - format based on mode
        filename_suffix = "_simple" if mode == "simple" else "_detailed"
        output_file = f"validation_results{filename_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Format results based on mode
        if mode == "simple":
            # Simple format: image name (without extension) -> {"result": "Yes/No"}
            simple_results = {}
            for image_name, result in results.items():
                # Remove file extension from image name
                clean_name = os.path.splitext(image_name)[0]
                simple_results[clean_name] = {"result": result["result"]}
            with open(output_file, 'w') as f:
                json.dump(simple_results, f, indent=2)
        else:
            # Detailed format: full information
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        if "error" not in results:
            total = len(results)
            # Results are always in detailed format internally
            passed = sum(1 for r in results.values() if r.get("result") == "Yes")
            failed = sum(1 for r in results.values() if r.get("result") == "No")
            errors = sum(1 for r in results.values() if r.get("result") == "Error")
            
            print(f"\nSummary:")
            print(f"  Total images: {total}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")
            print(f"  Errors: {errors}")
            
            if mode == "detailed" and failed > 0:
                print(f"\n💡 Check detailed failure reasons in {output_file}")
    
    else:
        print(f"Error: Path not found: {input_path}")

if __name__ == "__main__":
    main()
