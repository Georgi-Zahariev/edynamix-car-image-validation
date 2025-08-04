#!/usr/bin/env python3
"""
Version 3: AI Vision-based car image validation
Uses AI vision models (GPT-4 Vision or Gemini 2.5 Pro) for intelligent image analysis
"""

import os
import sys
import json
from datetime import datetime
from gpt_vision_checker import check_car_image

def get_user_choice():
    """Ask user for validation mode"""
    print("🚗 Car Image Validation - Version 3 (AI Vision)")
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

def validate_single_image(image_path: str, mode: str = "detailed") -> dict:
    """
    Validate a single car image using GPT Vision
    
    Args:
        image_path: Path to the image file
        mode: "simple" or "detailed"
        
    Returns:
        dict: Validation result with metadata
    """
    try:
        if not os.path.exists(image_path):
            return {
                "result": "Error",
                "error": "File not found",
                "timestamp": datetime.now().isoformat()
            }
        
        result = check_car_image(image_path)
        
        # Add metadata to the full result
        result["timestamp"] = datetime.now().isoformat()
        result["model_used"] = "gpt-4o"
        result["image_path"] = image_path
        
        # Mode only affects the final output format, not the validation itself
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
        
        result = validate_single_image(image_path, mode)
        results[image_file] = result
        
        # Print immediate result - same for both modes
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
        result = validate_single_image(input_path, mode)
        
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
