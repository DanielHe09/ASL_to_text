#!/bin/bash
# Clean, simple setup for ASL project
# This will get ONE thing working: signtalk.py (the simpler MediaPipe approach)

set -e  # Exit on error

echo "🧹 Cleaning up..."
cd "$(dirname "$0")"

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "Removing old venv..."
    rm -rf venv
fi

# Create fresh venv with Python 3.10
echo "📦 Creating fresh virtual environment..."
python3.10 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install ONLY what's needed for signtalk.py
echo "📥 Installing dependencies..."
pip install opencv-python mediapipe scikit-learn pyttsx3 numpy

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import cv2, mediapipe, sklearn; print('✅ All imports successful!')" || {
    echo "❌ Installation failed"
    exit 1
}

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Test webcam:   cd ASL_to_English && python webcam_smoke_test.py"
echo "  3. Run signtalk:  cd ASL_to_English && python signtalk.py --labels 'Hello,Thanks,Yes,No'"
echo ""

