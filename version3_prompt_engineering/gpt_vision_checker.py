# GPT Vision-based car image validation
# Note: While this code uses OpenAI's API, best results were achieved using 
# the same prompt with Google's Gemini 2.5 Pro model for superior accuracy.
from dotenv import load_dotenv
import os
import base64
import json
from openai import OpenAI
from prompts import DETAILED_CAR_VALIDATION_PROMPT

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

client = OpenAI(api_key=api_key)

def check_car_image(image_path: str) -> dict:
    """
    Analyze a car image using GPT-4 Vision to check if it meets validation criteria.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Full validation result with criteria breakdown
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Determine image format for proper base64 encoding
    image_ext = os.path.splitext(image_path)[1].lower()
    if image_ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif image_ext == '.png':
        mime_type = "image/png"
    elif image_ext == '.webp':
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default
    
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    # Use detailed prompt for comprehensive validation
    prompt = DETAILED_CAR_VALIDATION_PROMPT

    # Call GPT-4o Vision
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=800,
        temperature=0.0  # Zero temperature for consistency
    )

    # Extract and parse model output
    raw_text = response.choices[0].message.content.strip()
    
    # Clean up markdown code blocks if present
    if raw_text.startswith('```json'):
        raw_text = raw_text[7:]  # Remove ```json
    if raw_text.endswith('```'):
        raw_text = raw_text[:-3]  # Remove ```
    raw_text = raw_text.strip()
    
    try:
        result = json.loads(raw_text)
        
        # Ensure backward compatibility - always return full detailed result
        if "criteria" not in result and result.get("result") in ["Yes", "No"]:
            result["criteria"] = {
                "contains_car": result["result"] == "Yes",
                "side_view": result["result"] == "Yes", 
                "white_background": result["result"] == "Yes",
                "proper_size": result["result"] == "Yes",
                "correct_orientation": result["result"] == "Yes"
            }
            if result["result"] == "No":
                result["failure_reasons"] = ["Detailed analysis not available"]
        
        return result
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "result": "No",
            "criteria": {
                "contains_car": False,
                "side_view": False,
                "white_background": False,
                "proper_size": False,
                "correct_orientation": False
            },
            "failure_reasons": ["Failed to parse GPT response"],
            "error": "JSON parsing failed"
        }

if __name__ == "__main__":
    # Test the validation function
    print("Testing car image validation...")
    
    for img in ["image1.png", "image2.png"]:
        img_path = f"../assets/test_images/{img}"
        try:
            result = check_car_image(img_path)
            print(f"\n{img}: {result['result']}")
            
            # Show detailed criteria
            if "criteria" in result:
                print("  Criteria check:")
                for criterion, passed in result["criteria"].items():
                    status = "✅" if passed else "❌"
                    print(f"    {criterion}: {status}")
            
            # Show failure reasons if available
            if result.get("result") == "No" and "failure_reasons" in result:
                print("  Failure reasons:")
                for reason in result["failure_reasons"]:
                    print(f"    • {reason}")
                    
        except FileNotFoundError:
            print(f"{img}: File not found")
        except Exception as e:
            print(f"{img}: Error - {e}")
