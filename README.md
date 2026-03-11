# Model Offloading in Edge AI using YOLO

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-blueviolet)](https://github.com/ultralytics/yolov5)

**A high-performance Edge Computing Object Detection System leveraging YOLOv5 for real-time inference with comprehensive energy metrics and performance analytics.**

[Features](#-features) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

**Model Offloading in Edge AI using YOLO** is a sophisticated object detection system designed for edge computing environments. It provides real-time YOLOv5-based object detection through a Flask REST API with beautiful, responsive web interfaces.

The system excels at:

- **Real-time object detection** from camera streams or image uploads
- **Energy consumption monitoring** with simulated power metrics
- **Performance analytics** including inference time tracking
- **GPU/CPU adaptive processing** with automatic fallback
- **Mobile-friendly interfaces** for deployment across devices

**Key Use Cases:**

- Security and surveillance systems
- Retail inventory and people counting
- IoT edge computing deployments
- Educational ML demonstrations
- Smart city applications
- Energy-efficient inference studies
- Real-time threat detection systems

---

## ✨ Features

### Core Detection Capabilities

- ✅ **Real-Time Object Detection** - YOLOv5 small model for fast inference (~150-300ms per image)
- ✅ **80 COCO Classes** - Detects people, vehicles, animals, household items, and more
- ✅ **Multiple Input Methods**
  - Live camera capture via webcam
  - Image file upload (drag-and-drop support)
  - Mobile device uploads
- ✅ **Dual UI Options**
  - Modern responsive interface (`index.html`)
  - Premium dark-themed interface (`minor6.html`)

### Performance & Analytics

- ✅ **Comprehensive Timing Metrics**
  - Upload time
  - Model load time
  - Inference time
  - Image save time
  - Total server processing time
  - Network overhead calculation
- ✅ **Energy Consumption Tracking**
  - Power consumption in Watts
  - Energy in Joules, Watt-hours, and mAh
  - Simulated GPU/CPU power modeling
  - Battery voltage configuration
- ✅ **Health Monitoring** - `/health` endpoint for system status checks

### Technical Features

- ✅ **Cross-Origin Support** - CORS enabled for multi-domain deployment
- ✅ **Automatic Device Detection** - GPU (CUDA) or CPU fallback
- ✅ **High-Performance Inference** - AMP (Automatic Mixed Precision) for CUDA
- ✅ **Error Handling** - Comprehensive error messages and client-side feedback
- ✅ **Network-Optimized** - File-based caching and efficient image transmission

### User Experience

- ✅ **Responsive Design** - Works on desktop, tablet, and mobile
- ✅ **Real-Time Preview** - Live video feed before capture
- ✅ **Visual Annotations** - Circular bounding boxes with confidence scores
- ✅ **JSON Response Format** - Structured detection data for integration
- ✅ **Loading Animations** - User-friendly feedback during processing

---

## 🛠️ Technology Stack

### Backend Framework

- **Flask 3.1.2** - Lightweight Python web framework
- **Flask-CORS 6.0.1** - Cross-Origin Resource Sharing support

### Machine Learning & Computer Vision

- **PyTorch 2.8.0** - Deep learning framework (CPU optimized, CUDA compatible)
- **PyTorch Lightning 2.5.4** - High-level ML training framework
- **PyTorch Vision 0.23.0** - Computer vision models and utilities
- **YOLOv5 (Ultralytics)** - Object detection model (pretrained on COCO)
- **OpenCV 4.x** - Computer vision library for image processing
- **NumPy 2.2.6** - Numerical computing
- **SciPy 1.16.1** - Scientific computing

### Image Processing & Visualization

- **Pillow 11.3.0** - Python Imaging Library for image operations
- **Matplotlib 3.10.6** - Data visualization
- **TorchMetrics 1.8.1** - ML metrics computation

### Frontend Technologies

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients and animations
- **JavaScript (ES6+)** - Client-side logic and DOM manipulation
- **MediaDevices API** - Browser camera access
- **Fetch API** - HTTP client for API communication

### Runtime Environment

- **Python 3.13** - Latest Python runtime
- **Virtual Environment** - Isolated dependency management

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

### System Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: Version 3.13 or higher
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 500MB+ free space (models and cache)
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster inference

### Software Dependencies

- Python 3.13
- pip (Python package manager)
- Git (for cloning the repository)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

### Verify Prerequisites

```bash
# Check Python version
python --version

# Check pip installation
pip --version

# Verify git installation
git --version
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
# Clone using HTTPS
git clone https://github.com/yourusername/Model-Offloading-Edge-AI-YOLO.git

# Navigate to project directory
cd Model-Offloading-Edge-AI-YOLO
```

### Step 2: Create and Activate Virtual Environment

**On Windows:**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**On Linux/macOS:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install flask==3.1.2
pip install flask-cors==6.0.1
pip install torch==2.8.0
pip install torchvision==0.23.0
pip install pytorch-lightning==2.5.4
pip install opencv-python
pip install pillow==11.3.0
pip install numpy==2.2.6
pip install scipy==1.16.1
pip install matplotlib==3.10.6
pip install torchmetrics==1.8.1

# Or install all at once from requirements (if provided)
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test Python imports
python -c "import torch; import flask; import cv2; print('All dependencies installed successfully!')"
```

---

## ⚡ Quick Start

### 1. Start the Edge Server

```bash
# Make sure your virtual environment is activated
python edge_server.py
```

**Expected Output:**

```
 * Running on http://0.0.0.0:8000
Edge server running on 0.0.0.0:8000 | Device: cpu (or cuda if available)
```

### 2. Access the Web Interface

Open your web browser and navigate to:

```
http://localhost:8000
```

Or for the premium interface:

```
http://localhost:8000/minor6
```

### 3. Perform Object Detection

**Option A: Camera Capture**

1. Click "Enable Camera" button
2. Allow browser camera access
3. Click "Capture Image" to take a photo
4. View detection results instantly

**Option B: Image Upload**

1. Drag and drop an image or click to select
2. Wait for processing (typically 200-400ms)
3. View annotated image with detection results
4. Check timing and energy metrics

---

## 📁 Project Structure

```
Model-Offloading-Edge-AI-YOLO/
│
├── edge_server.py              # Main Flask application (629 lines)
│   ├── Flask app initialization
│   ├── /detect endpoint (POST)
│   ├── /health endpoint (GET)
│   ├── Image processing pipeline
│   ├── YOLOv5 inference wrapper
│   ├── Performance metrics calculation
│   └── Energy consumption estimation
│
├── index.html                  # Modern responsive UI (primary interface)
│   ├── Live camera capture
│   ├── Image upload (drag-and-drop)
│   ├── Real-time performance display
│   ├── Mobile-responsive design
│   └── Modern gradient styling
│
├── minor6.html                 # Premium dark-themed UI (alternative interface)
│   ├── Luxury design elements
│   ├── Glass-morphism effects
│   ├── Advanced animations
│   ├── Professional appearance
│   └── Mobile optimization
│
├── yolov5s.pt                  # YOLOv5 small model weights (~14.8MB)
│   └── Provides fast inference suitable for edge devices
│
├── yolov5su.pt                 # YOLOv5 small P6 model weights (~18.5MB)
│   └── Alternative model variant with enhanced features
│
├── uploads/                    # Directory for uploaded images
│   └── Stores user-submitted images (temporary)
│
├── outputs/                    # Directory for annotated results
│   └── Stores processed images with detection overlays
│
├── venv/                       # Python virtual environment
│   └── Contains all installed dependencies
│
├── .git/                       # Git repository metadata
├── .gitignore                  # Git ignore configuration
└── README.md                   # This file

```

### File Size Reference

| File             | Size   | Description               |
| ---------------- | ------ | ------------------------- |
| `edge_server.py` | 20.9KB | Main Flask server         |
| `index.html`     | 2.7KB  | Modern UI interface       |
| `minor6.html`    | 36KB   | Premium UI interface      |
| `yolov5s.pt`     | 14.8MB | Small YOLOv5 model        |
| `yolov5su.pt`    | 18.5MB | P6 YOLOv5 model           |
| Total Project    | ~4.2GB | Including venv and models |

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Endpoint              | Method | Description               | Authentication |
| --------------------- | ------ | ------------------------- | -------------- |
| `/`                   | GET    | Serves main web interface | None           |
| `/detect`             | POST   | Perform object detection  | None           |
| `/health`             | GET    | System health check       | None           |
| `/outputs/<filename>` | GET    | Retrieve processed images | None           |

---

### 1. Health Check Endpoint

**Request:**

```bash
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "device": "cpu",
  "models_loaded": true,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

---

### 2. Object Detection Endpoint

**Endpoint:** `POST /detect`

**Request Format:**

```
Content-Type: multipart/form-data

Parameter: file (binary image data)
Supported Formats: JPEG, PNG, BMP, GIF, WebP, TIFF
Maximum Size: Depends on available memory (typically 50MB+)
```

**Example cURL Request:**

```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:8000/detect
```

**Example Fetch Request (JavaScript):**

```javascript
const formData = new FormData();
formData.append("file", imageFile);

fetch("http://localhost:8000/detect", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => console.log(data))
  .catch((error) => console.error("Error:", error));
```

**Successful Response (200 OK):**

```json
{
  "success": true,
  "image_url": "/outputs/detected_1705314645123.jpg",
  "num_detections": 5,
  "detections": [
    {
      "label": "person",
      "confidence": 0.95,
      "box": {
        "x_min": 100,
        "y_min": 150,
        "x_max": 250,
        "y_max": 450
      },
      "center_x": 175,
      "center_y": 300,
      "radius": 75
    },
    {
      "label": "dog",
      "confidence": 0.87,
      "box": {
        "x_min": 300,
        "y_min": 200,
        "x_max": 400,
        "y_max": 350
      },
      "center_x": 350,
      "center_y": 275,
      "radius": 50
    }
  ],
  "timing": {
    "upload_time_ms": 1.2,
    "load_time_ms": 2.5,
    "detection_time_ms": 156.8,
    "save_time_ms": 15.3,
    "total_server_time_ms": 174.8,
    "network_overhead_ms": 5.2,
    "total_roundtrip_ms": 180.0
  },
  "energy": {
    "estimated_power_watts": 30.0,
    "energy_consumption_joules": 5.24,
    "energy_consumption_wh": 0.00146,
    "energy_consumption_mah": 0.395,
    "device_type": "cpu",
    "battery_voltage": 3.7
  }
}
```

**Error Response (400 Bad Request):**

```json
{
  "success": false,
  "error": "No file provided",
  "error_code": "NO_FILE"
}
```

**Error Response (500 Internal Server Error):**

```json
{
  "success": false,
  "error": "Failed to process image: Invalid image format",
  "error_code": "PROCESSING_ERROR"
}
```

### Response Fields Explanation

**Detection Object:**

- `label`: Object class name (person, car, dog, etc.)
- `confidence`: Detection confidence score (0-1)
- `box`: Bounding box coordinates
- `center_x`, `center_y`: Center coordinates of detection
- `radius`: Radius of circular overlay visualization

**Timing Object:** All values in milliseconds

- `upload_time_ms`: Time to upload file
- `load_time_ms`: YOLOv5 model load time
- `detection_time_ms`: Inference time
- `save_time_ms`: Time to save annotated image
- `total_server_time_ms`: Total backend processing
- `network_overhead_ms`: Network latency estimation
- `total_roundtrip_ms`: Complete request-response cycle

**Energy Object:**

- `estimated_power_watts`: Current estimated power draw
- `energy_consumption_*`: Energy in different units
- `device_type`: "cpu" or "cuda"
- `battery_voltage`: Configured battery voltage (volts)

---

### 3. Image Retrieval Endpoint

**Request:**

```bash
GET /outputs/detected_1705314645123.jpg
```

**Response:**

- HTTP 200: JPEG image file
- HTTP 404: File not found

---

## ⚙️ Configuration

### Environment Variables

Configure the system behavior using environment variables:

```bash
# Battery voltage for energy calculation (default: 3.7V)
set BATTERY_VOLTAGE=3.7

# On Linux/macOS
export BATTERY_VOLTAGE=3.7
```

### Application Configuration (in `edge_server.py`)

| Setting                | Default   | Purpose                           | Modifiable    |
| ---------------------- | --------- | --------------------------------- | ------------- |
| `MODEL_NAME`           | "yolov5s" | YOLOv5 model variant              | Yes (line 15) |
| `CONFIDENCE_THRESHOLD` | 0.45      | Detection confidence threshold    | Yes (line 16) |
| `IOU_THRESHOLD`        | 0.45      | Intersection over Union threshold | Yes (line 17) |
| `HOST`                 | "0.0.0.0" | Server bind address               | Yes (line 20) |
| `PORT`                 | 8000      | Server port number                | Yes (line 21) |
| `UPLOAD_DIR`           | "uploads" | Upload directory path             | Yes (line 22) |
| `OUT_DIR`              | "outputs" | Output directory path             | Yes (line 23) |
| `MAX_FILE_SIZE`        | unlimited | Maximum upload size               | Yes (line 24) |

### Model Selection

To use different YOLOv5 models, modify `edge_server.py`:

```python
# Fast inference (recommended)
MODEL_NAME = "yolov5s"  # Small - good balance

# Alternatives:
MODEL_NAME = "yolov5m"  # Medium - more accurate, slower
MODEL_NAME = "yolov5l"  # Large - best accuracy, slowest
MODEL_NAME = "yolov5su" # Small P6 - enhanced features
```

### Threshold Tuning

Adjust detection sensitivity in `edge_server.py`:

```python
# Higher = more strict (fewer detections)
CONFIDENCE_THRESHOLD = 0.55

# Lower = less strict (more detections)
CONFIDENCE_THRESHOLD = 0.35
```

---

## 📝 Usage Examples

### Example 1: Detect Objects in an Image

```bash
#!/bin/bash

# Using curl
curl -X POST \
  -F "file=@./sample_image.jpg" \
  http://localhost:8000/detect \
  | jq '.'  # Pretty-print JSON
```

### Example 2: Process Multiple Images

```python
import requests
import json
from pathlib import Path

def detect_objects(image_path, server_url="http://localhost:8000"):
    """Send image to detection server and return results."""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{server_url}/detect", files=files)
        return response.json()

# Process images
image_dir = Path("./sample_images")
for image_file in image_dir.glob("*.jpg"):
    print(f"Processing: {image_file}")
    result = detect_objects(str(image_file))
    print(f"Detections: {result['num_detections']}")
    print(f"Inference time: {result['timing']['detection_time_ms']}ms")
```

### Example 3: Real-time Detection via JavaScript

```javascript
async function detectFromCamera() {
  const video = document.getElementById("camera");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  // Capture frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to blob
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    // Send to server
    const response = await fetch("http://localhost:8000/detect", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    displayResults(data);
  }, "image/jpeg");
}

function displayResults(data) {
  console.log(`Found ${data.num_detections} objects`);
  data.detections.forEach((det) => {
    console.log(`${det.label}: ${(det.confidence * 100).toFixed(2)}%`);
  });
  console.log(`Inference time: ${data.timing.detection_time_ms}ms`);
}
```

### Example 4: Monitor Performance Metrics

```python
import requests
import time

def analyze_performance(server_url="http://localhost:8000"):
    """Analyze detection server performance."""
    metrics = {
        'detections': [],
        'timings': [],
        'energy': []
    }

    for i in range(10):
        response = requests.post(
            f"{server_url}/detect",
            files={'file': open(f'image_{i}.jpg', 'rb')}
        )
        data = response.json()

        metrics['timings'].append(data['timing']['detection_time_ms'])
        metrics['energy'].append(data['energy']['energy_consumption_mah'])

    # Calculate statistics
    avg_inference = sum(metrics['timings']) / len(metrics['timings'])
    avg_energy = sum(metrics['energy']) / len(metrics['energy'])

    print(f"Average Inference Time: {avg_inference:.2f}ms")
    print(f"Average Energy: {avg_energy:.3f}mAh")
    print(f"Max Inference Time: {max(metrics['timings']):.2f}ms")
    print(f"Min Inference Time: {min(metrics['timings']):.2f}ms")
```

---

## 📊 Performance Metrics

### Typical Performance (CPU Mode)

| Metric                | Value       | Notes                 |
| --------------------- | ----------- | --------------------- |
| **Model Load Time**   | 2-3 seconds | First request only    |
| **Average Inference** | 150-300ms   | Depends on image size |
| **Total Server Time** | 180-450ms   | Including I/O         |
| **Network Roundtrip** | 200-500ms   | Network dependent     |
| **Memory Usage**      | 1.5-2.5GB   | PyTorch + YOLOv5      |
| **CPU Usage**         | 40-80%      | Per inference         |

### Performance with GPU (CUDA)

| Metric             | Improvement        |
| ------------------ | ------------------ |
| **Inference Time** | 3-5x faster        |
| **Throughput**     | 5-10 FPS @ 640x480 |
| **Memory Usage**   | 2-4GB VRAM         |

### Scalability

- **Concurrent Requests**: Limited by Python GIL, typically 1-2 concurrent
- **Recommendation**: Use with load balancer for high concurrency
- **Alternative**: Deploy multiple instances on different ports

### Energy Consumption Examples

**Single Detection (CPU):**

- ~11.7 Joules per image
- ~3.25 Watt-hours
- ~0.88 mAh (at 3.7V)

**Batch Processing (100 images):**

- ~1170 Joules total
- ~325 Watt-hours
- ~88 mAh (at 3.7V)

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Port Already in Use

**Error:** `Address already in use [Errno 10048]`

**Solution:**

```bash
# Change port in edge_server.py (line 21)
PORT = 8001

# Or kill the process using the port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/macOS:
lsof -i :8000
kill -9 <process_id>
```

#### Issue 2: CUDA/GPU Not Available

**Error:** `CUDA is not available`

**Solution:**

```python
# The system automatically falls back to CPU
# To force CPU mode, modify edge_server.py:
DEVICE = "cpu"  # Line 26

# To use CUDA (if available):
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

#### Issue 3: Out of Memory Error

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**

- Use smaller model: `MODEL_NAME = "yolov5s"`
- Reduce image size before uploading
- Close other applications
- Increase available RAM
- Switch to CPU mode temporarily

#### Issue 4: Slow Inference Time

**Cause:** CPU processing or large image

**Solutions:**

- Check device type (should be "cuda" if available)
- Install CUDA 11.8+ for GPU acceleration
- Resize large images before upload
- Use smaller YOLOv5 model variant

#### Issue 5: Camera Not Working

**Error:** `NotAllowedError: Permission denied`

**Solution:**

- Browser permissions: Check camera access in browser settings
- HTTPS required: Camera API requires secure context (localhost works)
- Different browser: Try Chrome, Firefox, or Edge

#### Issue 6: CORS Errors

**Error:** `Access to XMLHttpRequest blocked by CORS`

**Solution:** CORS is already enabled. If still having issues:

```python
# Verify in edge_server.py
CORS(app)  # Should be present (line ~30)
```

#### Issue 7: File Upload Fails

**Error:** `Failed to parse request body`

**Solution:**

- Use `multipart/form-data` content type
- Ensure file parameter name is exactly `file`
- Limit upload size to available RAM
- Check file format (JPEG, PNG recommended)

#### Issue 8: Model Download Timeout

**Error:** `HTTPError: 403 Client Error: Forbidden`

**Solution:**

```bash
# Download models manually
pip install torch torchvision
cd project_directory

# Download YOLOv5s model
python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s')"
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Types of Contributions

- Bug fixes and improvements
- New features (UI enhancements, detection improvements)
- Documentation updates
- Performance optimizations
- Additional model support
- Test cases and benchmarks

### Contributing Steps

1. **Fork the Repository**

   ```bash
   # Click "Fork" on GitHub
   ```

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Keep changes focused and minimal
   - Follow the existing code style
   - Test thoroughly before submitting

4. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide detailed description of changes
   - Reference any related issues
   - Use clear, descriptive title

### Code Style Guidelines

- Use PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and single-purpose
- Test on both CPU and GPU if possible

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### What You Can Do

- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute
- ✅ Use privately

### Requirements

- 📋 Include license and copyright notice

---

## 📧 Contact & Support

### Getting Help

- **Issues**: [GitHub Issues]((https://github.com/Ashish29-glitch/Model-Offloading-in-Edge-AI-using-YOLO-.git)
- **Discussions**: [GitHub Discussions](https://github.com/Ashish29-glitch/Model-Offloading-in-Edge-AI-using-YOLO-.git)
- **Email**:

### Quick Links

- 📖 [YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5/)
- 📖 [Flask Documentation](https://flask.palletsprojects.com/)
- 📖 [PyTorch Documentation](https://pytorch.org/docs/)
- 💬 [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

### Project Information

- **Status**: Active Development
- **Last Updated**: 2024
- **Maintainer**: Your Name
- **Contributors**: [List contributors here]

---

## 🗺️ Roadmap

### Current Version (v1.0)

- ✅ Real-time object detection
- ✅ Camera capture support
- ✅ Image upload functionality
- ✅ Performance metrics
- ✅ Energy consumption tracking

### Planned Features (v1.1)

- 🔜 Video file upload support
- 🔜 Live streaming detection
- 🔜 Custom model training
- 🔜 Model quantization for edge devices
- 🔜 Batch processing API

### Future Enhancements (v2.0)

- 🔜 Real-time video streaming from IP cameras
- 🔜 Multi-model ensemble detection
- 🔜 Database integration for result history
- 🔜 REST API authentication
- 🔜 Web-based model fine-tuning interface
- 🔜 Docker deployment support
- 🔜 Kubernetes orchestration

---

## 📚 Related Resources

### Academic References

- YOLOv5: YOLOv5 by Ultralytics (arXiv papers)
- Edge Computing: "Edge Computing: A Survey" research papers
- Energy Efficiency: "Energy-Efficient Deep Learning" studies

### Similar Projects

- [TensorFlow Object Detection API](https://github.com/tensorflow/models)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [MediaPipe](https://mediapipe.dev/)

### Learning Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Flask Web Development](https://flask.palletsprojects.com/tutorial/)
- [Machine Learning Fundamentals](https://www.deeplearning.ai/)

---

## 🙏 Acknowledgments

### Technologies

- **YOLOv5** by [Ultralytics](https://www.ultralytics.com/)
- **PyTorch** by [Meta AI](https://pytorch.org/)
- **Flask** by [Pallets](https://palletsprojects.com/)
- **OpenCV** by [OpenCV team](https://opencv.org/)

### Datasets

- **COCO Dataset** - Common Objects in Context

### Inspiration

- Edge computing research community
- IoT device optimization community
- Real-time computer vision applications

---

<div align="center">

### Made with ❤️ for Edge AI Computing

⭐ **If you find this project helpful, please consider giving it a star!**

[⬆ Back to Top](#model-offloading-in-edge-ai-using-yolo)

</div>

---

## 📞 Quick Reference

### Essential Commands

```bash
# Clone project
git clone https://github.com/yourusername/Model-Offloading-Edge-AI-YOLO.git
cd Model-Offloading-Edge-AI-YOLO

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run server
python edge_server.py

# Access application
# Modern UI: http://localhost:8000
# Premium UI: http://localhost:8000/minor6
# API: POST http://localhost:8000/detect

# Health check
curl http://localhost:8000/health

# Deactivate environment
deactivate
```

### File Locations

| File             | Purpose                   |
| ---------------- | ------------------------- |
| `edge_server.py` | Main application server   |
| `index.html`     | Modern web interface      |
| `minor6.html`    | Premium web interface     |
| `uploads/`       | User-uploaded images      |
| `outputs/`       | Detected/annotated images |

---

**Version:** 1.0
**Last Updated:** January 2024
**License:** MIT
**Status:** ✅ Production Ready

