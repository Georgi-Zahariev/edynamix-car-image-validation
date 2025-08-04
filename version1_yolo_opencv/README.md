# Version 1: YOLO + OpenCV Car Image Validation

## Overview

This version implements a **traditional computer vision approach** using YOLOv8 for car detection combined with OpenCV for image analysis. It serves as a baseline implementation using established CV techniques.

## ⚠️ Why This Approach Has Limitations

While functional, this approach has several inherent weaknesses:

1. **Rigid Rule-Based Logic**: Uses hard-coded thresholds and simple heuristics that don't adapt to edge cases
2. **Limited Orientation Detection**: Basic position-based heuristics are unreliable for determining car orientation
3. **Simplistic Background Analysis**: Basic color averaging can be fooled by shadows, lighting, or non-uniform backgrounds
4. **No Context Understanding**: Cannot distinguish between similar-looking scenarios that require different judgments
5. **Maintenance Overhead**: Requires manual tuning of thresholds for different image datasets

**Note**: This is a baseline implementation using traditional computer vision techniques.

## Features

### Core Validation Criteria
- ✅ **Car Detection**: Uses YOLOv8n model for car detection (confidence > 0.5)
- ✅ **Size Check**: Car must occupy at least 25% of image area
- ✅ **Side View**: Car width-to-height ratio must be ≥ 1.8
- ✅ **White Background**: Background pixels must have RGB values > 180
- ✅ **Orientation**: Simple position-based left-facing detection

### Output Modes
- **Simple Mode**: Clean JSON with `{"image": {"result": "Yes/No"}}`
- **Detailed Mode**: Full criteria breakdown with failure reasons

## Technical Implementation

### Architecture
```
Input Image → YOLOv8 Detection → OpenCV Analysis → Rule-Based Validation
```

### Key Components

1. **YOLO Detection** (`yolov8_loader.py`)
   - Uses YOLOv8n for car detection
   - Filters for COCO class 2 (cars only)
   - Confidence threshold: 0.5

2. **CV Analysis** (`utils.py`)
   - Size ratio calculation
   - Background color analysis
   - Aspect ratio checking
   - Position-based orientation heuristics

3. **Main Validator** (`main.py`)
   - Coordinates validation pipeline
   - Handles both single and batch processing
   - Generates timestamped output files

## Requirements

```bash
pip install ultralytics>=8.0.0 opencv-python numpy
```

## Usage

### Interactive Mode
```bash
python main.py
# Choose validation mode:
# 1. Simple mode (result: Yes/No only)  
# 2. Detailed mode (full criteria breakdown)
```

### Single Image
```bash
python main.py path/to/image.jpg
```

### Batch Processing
```bash
python main.py path/to/image/directory/
```

## Sample Output

### Simple Mode
```json
{
  "image1": {"result": "Yes"},
  "image2": {"result": "No"}
}
```

### Detailed Mode
```json
{
  "image1.png": {
    "result": "Yes",
    "criteria": {
      "contains_car": true,
      "side_view": true,
      "white_background": true,
      "proper_size": true,
      "correct_orientation": true
    },
    "timestamp": "2025-08-04T10:30:00.123456",
    "model_used": "yolov8n + opencv"
  }
}
```

## Validation Criteria Details

| Criterion | Method | Threshold | Limitations |
|-----------|---------|-----------|-------------|
| **Car Detection** | YOLOv8n COCO class 2 | Confidence > 0.5 | May miss unusual car angles |
| **Size Check** | Bounding box area ratio | ≥ 25% of image | Fixed threshold doesn't adapt |
| **Side View** | Width/height aspect ratio | ≥ 1.8 | Ignores actual car orientation |
| **Background** | RGB color averaging | All channels > 180 | Fails with shadows/gradients |
| **Orientation** | Car center position | 30%-80% from left | Very unreliable heuristic |

## Known Issues

1. **False Positives**: Cars at slight angles may pass as "side view"
2. **Background Sensitivity**: Shadows or gradients can cause failures
3. **Orientation Guessing**: Position-based orientation is highly unreliable
4. **Fixed Thresholds**: No adaptation to different image characteristics

## Files

- `main.py` - Main validation script with interactive mode
- `yolov8_loader.py` - YOLOv8 car detection wrapper
- `utils.py` - OpenCV-based validation functions
- `yolov8n.pt` - YOLOv8 nano model file
- `README.md` - This documentation

---

**Ready to use**: Run `python main.py` to get started with interactive validation.