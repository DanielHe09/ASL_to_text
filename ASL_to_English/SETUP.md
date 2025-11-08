# Setup Guide for ASL to English Translator

This guide will help you set up the Python environment and dependencies needed to run this project.

## Prerequisites

- Python 3.7-3.10 (Python 3.11+ may have compatibility issues)
- pip (Python package manager)
- Git (already done if you cloned the repo)
- Webcam (for real-time detection)

## Step 1: Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to avoid conflicts with other projects:

```bash
# Navigate to the project directory
cd ASL_to_English/ASL_to_English

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

## Step 2: Install System Dependencies

### macOS (using Homebrew):
```bash
brew install protobuf
```

### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install protobuf-compiler
```

### Windows:
Download and install protoc from: https://github.com/protocolbuffers/protobuf/releases
Add it to your PATH.

## Step 3: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow 2.7.0 (which is no longer available), the requirements.txt uses TensorFlow 2.8.0+ which should work. If you need a specific version, you can modify requirements.txt.

## Step 4: Set Up TensorFlow Object Detection API

The notebook will clone the TensorFlow models repository if it doesn't exist, but you need to compile the protobuf files and install the API:

```bash
# Navigate to the models research directory
cd Tensorflow/models/research

# Compile protobuf files
protoc object_detection/protos/*.proto --python_out=.

# Install the object detection package
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## Step 5: Install TF Slim (if needed)

```bash
cd Tensorflow/models/research/slim
pip install -e .
```

## Step 6: Set Up Environment Variables (Optional but Recommended)

Add the following to your shell profile (`~/.bashrc`, `~/.zshrc`, or Windows environment variables):

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Tensorflow/models/research:$(pwd)/Tensorflow/models/research/slim"
```

Or add this at the beginning of your notebook:

```python
import sys
import os
sys.path.append('Tensorflow/models/research')
sys.path.append('Tensorflow/models/research/slim')
```

## Step 7: Verify Installation

Run this in Python to verify everything is installed:

```python
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print("All imports successful!")
```

## Step 8: Run the Notebook

1. Open Jupyter Notebook or JupyterLab:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

2. Open `Signlangtranslator.ipynb`

3. Run the cells in order. The notebook will:
   - Set up directories
   - Clone TensorFlow models (if needed)
   - Install dependencies
   - Load the trained model
   - Run real-time detection

## Troubleshooting

### Issue: "No module named 'object_detection'"
**Solution:** Make sure you've completed Step 4 and added the paths to PYTHONPATH.

### Issue: "protoc: command not found"
**Solution:** Install protobuf compiler (Step 2) and ensure it's in your PATH.

### Issue: TensorFlow version conflicts
**Solution:** The requirements.txt uses TensorFlow 2.8.0+. If you need a specific version, modify requirements.txt. Note that TensorFlow 2.7.0 is no longer available on PyPI.

### Issue: OpenCV webcam not working
**Solution:** 
- Make sure you have `opencv-python` installed (not just `opencv-python-headless`)
- On macOS, you may need to grant camera permissions
- Try different camera indices (0, 1, 2) in `cv2.VideoCapture(0)`

### Issue: CUDA/GPU errors
**Solution:** If you don't have a compatible GPU, TensorFlow will use CPU (slower but works). For GPU support, install `tensorflow-gpu` instead.

## Model Checkpoints

The repository should include trained model checkpoints in `my_ssd_mobnet/`. If they're missing, you'll need to:
1. Collect training images using `realtime_image_collection.ipynb`
2. Annotate them using LabelImg
3. Train the model (follow the notebook steps)

## Next Steps

1. Test the real-time detection (Section 10 of the notebook)
2. If the model isn't trained yet, collect and annotate images first
3. Adjust confidence thresholds in the detection code if needed


