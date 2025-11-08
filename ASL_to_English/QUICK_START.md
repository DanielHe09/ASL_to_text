# Quick Start Guide - Get ASL Project Working

## Current Status ✅
- ✅ Virtual environment created
- ✅ Trained model checkpoints exist (ckpt-5 through ckpt-11)
- ❌ Dependencies not installed yet
- ❌ TensorFlow Object Detection API not set up

## Option 1: Use the TensorFlow Object Detection Approach (Notebook)

This is the original approach from `Signlangtranslator.ipynb`.

### Step 1: Activate Virtual Environment
```bash
cd ~/Desktop/ASL_to_English
source venv/bin/activate
```

### Step 2: Install Core Dependencies
```bash
cd ASL_to_English
pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib pillow protobuf
```

### Step 3: Install System Dependency (protobuf compiler)
```bash
# macOS
brew install protobuf

# Verify
protoc --version
```

### Step 4: Set Up TensorFlow Object Detection API
```bash
# Clone TensorFlow models (if not already done)
cd ~/Desktop/ASL_to_English
if [ ! -d "Tensorflow/models" ]; then
    git clone https://github.com/tensorflow/models.git Tensorflow/models
fi

# Compile protobuf files
cd Tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.

# Install the object detection package
cp object_detection/packages/tf2/setup.py .
pip install .
```

### Step 5: Set Python Path
Add to your notebook or create a `.env` file:
```python
import sys
import os
sys.path.append('Tensorflow/models/research')
sys.path.append('Tensorflow/models/research/slim')
```

### Step 6: Test the Setup
```python
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
print("✅ Setup successful!")
```

### Step 7: Run Real-Time Detection
Open `Signlangtranslator.ipynb` and run Section 10 (Real Time Detections from your Webcam).

---

## Option 2: Use the Simpler MediaPipe Approach (signtalk.py)

This is a simpler alternative that doesn't require TensorFlow Object Detection API.

### Step 1: Activate Virtual Environment
```bash
cd ~/Desktop/ASL_to_English
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
cd ASL_to_English
pip install opencv-python mediapipe scikit-learn pyttsx3 numpy
```

### Step 3: Run the Application
```bash
python signtalk.py --labels "Hello,Thanks,Yes,No,Please,ILoveYou"
```

### How it Works:
- Uses MediaPipe Hands for hand tracking
- Train on-the-fly by pressing keys (1-6 for labels, 'c' to capture)
- Simpler setup, no pre-trained model needed
- See `signtalk.py` for full instructions

---

## Recommended: Start with Option 2 (Simpler)

If you want to get something working quickly, start with Option 2 (signtalk.py). It's much simpler and doesn't require the complex TensorFlow Object Detection API setup.

---

## Troubleshooting

### If TensorFlow installation fails:
```bash
# Try a specific version
pip install tensorflow==2.13.0
```

### If protoc command not found:
- macOS: `brew install protobuf`
- Linux: `sudo apt-get install protobuf-compiler`
- Windows: Download from https://github.com/protocolbuffers/protobuf/releases

### If webcam doesn't work:
```bash
# Test webcam first
python webcam_smoke_test.py
```

### If you get import errors:
Make sure you're in the venv:
```bash
which python  # Should show venv/bin/python
```

