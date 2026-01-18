import os, time, uuid
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
from PIL import Image
import torch
from flask import render_template_string
import cv2
import numpy as np

# Initialize Flask app first
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from mobile

# Simple mobile client page
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Edge Device - YOLOv5 Offloading</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body { 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        h2 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.2em;
            font-weight: 600;
            position: relative;
            padding-bottom: 15px;
        }

        h2:after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #007BFF, #00E5FF);
            border-radius: 2px;
        }

        .option-buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
        }

        button {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .camera-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
        }

        .upload-btn {
            background: linear-gradient(135deg, #007BFF, #00E5FF);
            color: white;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }

        .camera-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin: 20px auto;
            text-align: center;
        }

        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 15px;
            margin: 0 auto 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        #uploadSection {
            text-align: center;
            padding: 40px;
            border: 3px dashed #dee2e6;
            border-radius: 15px;
            margin: 20px 0;
            transition: all 0.3s ease;
        }

        #uploadSection:hover {
            border-color: #007BFF;
            background: #f8f9fa;
        }

        input[type=file] {
            margin: 20px 0;
            padding: 10px;
            width: 100%;
            max-width: 400px;
        }

        .result-container {
            margin-top: 30px;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .timing-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }

        .timing-info h4 {
            color: #2c3e50;
            margin-bottom: 15px;
        }

        .timing-info p {
            margin: 8px 0;
            color: #6c757d;
        }

        .image-container {
            margin: 25px 0;
        }

        .image-container img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        pre {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 14px;
            color: #2c3e50;
        }

        .loading {
            color: #007BFF;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .loading:after {
            content: '';
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007BFF;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }

            h2 {
                font-size: 1.8em;
            }

            .option-buttons {
                flex-direction: column;
            }

            button {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Edge Computing: YOLOv5 Detection</h2>
        
        <div class="option-buttons">
            <button id="showCamera" class="camera-btn">
                📸 Use Camera
            </button>
            <button id="showUpload" class="upload-btn">
                📁 Upload Photo
            </button>
        </div>

        <div id="cameraSection" class="camera-container" style="display: none;">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
            <div class="camera-controls">
                <button id="capture" class="camera-btn">
                    📸 Take Photo
                </button>
                <button id="closeCamera" class="camera-btn" style="background: linear-gradient(135deg, #dc3545, #ff6b6b);">
                    ✖ Close Camera
                </button>
            </div>
        </div>

        <div id="uploadSection" style="display: none;">
            <form id="uploadForm" enctype="multipart/form-data">
                <h3>Drop your image here or click to upload</h3>
                <input type="file" name="file" accept="image/*"><br>
                <button type="submit" class="upload-btn">Upload & Detect</button>
            </form>
        </div>

        <div id="result"></div>
    </div>

    <script>
    // Option buttons handling
    const showCameraBtn = document.getElementById('showCamera');
    const showUploadBtn = document.getElementById('showUpload');
    const cameraSection = document.getElementById('cameraSection');
    const uploadSection = document.getElementById('uploadSection');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    const closeCameraBtn = document.getElementById('closeCamera');
    const context = canvas.getContext('2d');
    
    let stream = null;

    // Show camera section
    showCameraBtn.addEventListener('click', async () => {
        uploadSection.style.display = 'none';
        cameraSection.style.display = 'block';
        
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                } 
            });
            video.srcObject = stream;
            video.style.display = 'block';
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Could not access camera. Please ensure camera permissions are granted.');
        }
    });

    // Show upload section
    showUploadBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        video.style.display = 'none';
        cameraSection.style.display = 'none';
        uploadSection.style.display = 'block';
    });

    // Close camera
    closeCameraBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        video.style.display = 'none';
        cameraSection.style.display = 'none';
    });

    // Capture photo
    captureButton.addEventListener('click', async () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'camera.jpg');
            
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            video.style.display = 'none';
            cameraSection.style.display = 'none';
            
            processImage(formData);
        }, 'image/jpeg', 0.95);
    });

    // Process image function (used by both camera and upload)
    async function processImage(formData) {
        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = '<p class="loading">Processing image...</p>';
        
        try {
            const startTime = performance.now();
            const res = await fetch("/detect", { 
                method: "POST", 
                body: formData 
            });
            const data = await res.json();
            
            if (!res.ok || !data.success) {
                const errMsg = data && data.error ? data.error : `Server error (${res.status})`;
                resultDiv.innerHTML = `<p style="color:red">Error: ${errMsg}</p>`;
                return;
            }
            
            const img = new Image();
            img.onload = () => {
                const endTime = performance.now();
                const totalTime = (endTime - startTime) / 1000;
                
                // Update the timing display in the HTML template
                resultDiv.innerHTML = `
    <div class="result-container">
        <h3>Detection Result ✅</h3>
        <div class="timing-info">
            <h4>Performance Metrics:</h4>
            <p>Upload Time: ${data.timing.upload_time} seconds</p>
            <p>Image Load Time: ${data.timing.load_time} seconds</p>
            <p>Detection Time: ${data.timing.detection_time} seconds</p>
            <p>Save Time: ${data.timing.save_time} seconds</p>
            <p>Total Server Processing: ${data.timing.total_server_time} seconds</p>
            <p>Network Overhead: ${(totalTime - data.timing.total_server_time).toFixed(3)} seconds</p>
            <p>Round Trip Time: ${totalTime.toFixed(3)} seconds</p>
            
            <h4>Energy Metrics:</h4>
            <p>Energy Consumption: ${data.timing.energy_consumption_joules.toFixed(3)} Joules</p>
            <p>Energy Usage: ${data.timing.energy_consumption_wh.toFixed(6)} Watt-hours</p>
            <p>Energy Usage: ${data.timing.energy_consumption_mah.toFixed(6)} mAh</p>

            <h4>Detected obj no.</h4>

            <!-- Added: number of objects detected -->
            <p>Number of Objects Detected: ${data.num_detections}</p>
        </div>
        <div class="image-container">
            <img src="${data.image_url}" alt="Annotated Image">
        </div>
        <h4>Detected Objects: (${data.num_detections})</h4>
        <pre>${JSON.stringify(data.detections, null, 2)}</pre>
    </div>
                `;
            };
            img.src = data.image_url;
            
        } catch (err) {
            console.error(err);
            resultDiv.innerHTML = '<p style="color:red">Error processing image</p>';
        }
    }

    // Handle file upload form
    const form = document.getElementById("uploadForm");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fileInput = form.querySelector('input[name="file"]');
        const file = fileInput.files[0];
        if (!file) return alert("Please select an image file");

        const formData = new FormData();
        formData.append("file", file);
        
        processImage(formData);
    });
    </script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML_PAGE)


# ----------- Config -----------
UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
HOST = "0.0.0.0"
PORT = 8000

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- Load YOLOv5 -----------
MODEL_NAME = "yolov5s"  # small & fast
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", MODEL_NAME, pretrained=True, verbose=False).to(device)
model.eval()

# ----------- Helper Functions -----------
def _save_pil(img: Image.Image, folder: str) -> str:
    name = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(folder, name)
    img.save(path, format="JPEG", quality=90)
    return path

def _results_to_dicts(results, img):
    dets = []
    names = results.names
    
    # Convert PIL Image to CV2 format
    cv_image = np.array(img)
    cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Get image dimensions
    height, width = cv_image.shape[:2]
    
    # Create a copy for drawing
    draw_image = cv_image.copy();

    if len(results.xyxy):
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            xmin, ymin, xmax, ymax = map(int, xyxy)
            
            # Calculate center and radius
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            radius = int(min(xmax - xmin, ymax - ymin) / 2)
            
            # Draw semi-transparent circle
            overlay = draw_image.copy()
            cv2.circle(overlay, (center_x, center_y), radius, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, draw_image, 0.7, 0, draw_image)
            
            # Draw circle border
            cv2.circle(draw_image, (center_x, center_y), radius, (0, 0, 255), 3)
            
            # Create label with confidence percentage
            label = f"{names[int(cls)]} ({conf:.1%})"
            
            # Adjust font size based on image dimensions
            font_scale = min(width, height) / 1000.0
            thickness = max(2, int(min(width, height) / 500))
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position text above circle
            text_x = center_x - text_width // 2
            text_y = center_y - radius - 10
            
            # Draw white background for text
            cv2.rectangle(draw_image,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (255, 255, 255),
                         -1)
            
            # Draw red text
            cv2.putText(draw_image,
                       label,
                       (text_x, text_y),
                       font,
                       font_scale,
                       (0, 0, 255),
                       thickness)

            dets.append({
                "label": names[int(cls)],
                "confidence": float(conf),
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius
            })

    # Convert back to PIL Image
    annotated_img = Image.fromarray(draw_image[:, :, ::-1])  # BGR to RGB
    return dets, annotated_img

# ----------- Routes -----------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": device}

@app.post("/detect")
def detect():
    timing = {}
    
    try:
        # Upload timing start
        timing['upload_start'] = time.perf_counter_ns()
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded", "success": False}), 400
        
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename", "success": False}), 400
            
        timing['upload_end'] = time.perf_counter_ns()

        # Image load timing
        timing['load_start'] = time.perf_counter_ns()
        img = Image.open(f.stream).convert("RGB")
        timing['load_end'] = time.perf_counter_ns()

        # Detection timing (includes model inference)
        timing['detection_start'] = time.perf_counter_ns()
        # Use appropriate context manager for CUDA autocast or no_grad
        if torch.cuda.is_available():
            ctx = torch.cuda.amp.autocast()
        else:
            ctx = torch.no_grad()
        with ctx:
            results = model(img, size=640)
            dets, annotated = _results_to_dicts(results, img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing['detection_end'] = time.perf_counter_ns()

        # Save timing
        timing['save_start'] = time.perf_counter_ns()
        timestamp = int(time.time())
        out_fname = f"detected_{timestamp}.jpg"
        out_path = os.path.join(OUT_DIR, out_fname)
        annotated.save(out_path, format="JPEG", quality=95, optimize=True)
        timing['save_end'] = time.perf_counter_ns()

        # Calculate final timings
        upload_time = (timing['upload_end'] - timing['upload_start']) / 1e9
        load_time = (timing['load_end'] - timing['load_start']) / 1e9
        detection_time = (timing['detection_end'] - timing['detection_start']) / 1e9
        save_time = (timing['save_end'] - timing['save_start']) / 1e9
        total_time = upload_time + load_time + detection_time + save_time

        # ---------- Energy Calculation ----------
        avg_power = 75 if device == "cuda" else 30  # in watts
        energy_joules = avg_power * detection_time
        energy_wh = energy_joules / 3600  # convert to Wh

        # convert Wh to mAh using a typical battery voltage (configurable via env)
        # mAh = (Wh / V) * 1000
        battery_voltage = float(os.getenv("BATTERY_VOLTAGE", 3.7))  # default 3.7V
        energy_mah = (energy_wh / battery_voltage) * 1000.0

        response_timing = {
            'upload_time': round(upload_time, 3),
            'load_time': round(load_time, 3),
            'detection_time': round(detection_time, 3),
            'save_time': round(save_time, 3),
            'total_server_time': round(total_time, 3),
            'energy_consumption_joules': round(energy_joules, 3),
            'energy_consumption_wh': round(energy_wh, 6),
            'energy_consumption_mah': round(energy_mah, 6)
        }
        
        
        # include number of detections in response
        return jsonify({
            "image_url": url_for('get_output_file', fname=out_fname, _external=True),
            "detections": dets,
            "num_detections": len(dets),
            "success": True,
            "timing": response_timing
        })

    except Exception as e:
        # Return the server error message so the client can display it
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.get("/outputs/<path:fname>")
def get_output_file(fname):
    """Serve output images"""
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(path, mimetype="image/jpeg", as_attachment=False)

if __name__ == "__main__":
    print(f"Edge server running on {HOST}:{PORT} | Device: {device}")
    app.run(host=HOST, port=PORT, debug=False)
# app.run(host=HOST, port=PORT, debug=True)