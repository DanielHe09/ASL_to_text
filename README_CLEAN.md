# ASL to English - Clean Setup Guide

## The Problem
This repo has multiple approaches, nested directories, and complex dependencies. Let's get ONE thing working simply.

## The Solution: Use signtalk.py (Simplest Option)

This uses MediaPipe (Google's hand tracking) - no TensorFlow Object Detection API needed.

## Quick Start (3 Steps)

### 1. Run the clean setup script
```bash
cd ~/Desktop/ASL_to_English
chmod +x CLEAN_SETUP.sh
./CLEAN_SETUP.sh
```

### 2. Activate the venv
```bash
source venv/bin/activate
```

### 3. Run it
```bash
cd ASL_to_English
python signtalk.py --labels "Hello,Thanks,Yes,No,Please,ILoveYou"
```

## What This Does

- **Creates a clean Python 3.10 virtual environment**
- **Installs only what's needed** (opencv, mediapipe, scikit-learn, etc.)
- **Uses the simpler MediaPipe approach** (no TensorFlow Object Detection API)
- **Trains on-the-fly** - you capture samples and it learns

## How to Use signtalk.py

1. **Run the script** (see above)
2. **Press number keys (1-6)** to select which sign you want to train
3. **Press 'c'** to capture a sample (when your hand is visible)
4. **Press 't'** to train the classifier
5. **Press 'r'** to start recognition
6. **Press 'q'** to quit

## Directory Structure (Simplified)

```
ASL_to_English/
├── venv/                    # Virtual environment (created by setup)
├── CLEAN_SETUP.sh          # Setup script
├── ASL_to_English/         # Main project folder
│   ├── signtalk.py         # ⭐ USE THIS (simple MediaPipe approach)
│   ├── webcam_smoke_test.py # Test your webcam
│   └── ... (other files, ignore for now)
```

## If You Want the TensorFlow Approach Later

The TensorFlow Object Detection approach (Signlangtranslator.ipynb) is more complex and requires:
- TensorFlow Object Detection API setup
- Protobuf compiler
- More dependencies

**Recommendation:** Get signtalk.py working first, then tackle the TensorFlow approach if needed.

## Troubleshooting

### "No module named cv2"
```bash
source venv/bin/activate
pip install opencv-python
```

### Webcam not working
```bash
python webcam_smoke_test.py  # Test which camera index works
```

### Wrong Python version
The setup script uses Python 3.10. If you don't have it:
```bash
# Check what you have
python3 --version
python3.9 --version
python3.10 --version

# Use whatever you have (3.9 or 3.10)
python3.9 -m venv venv  # or python3.10
```

