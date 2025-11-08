#!/bin/bash
# Quick setup script for ASL project

echo "🚀 Setting up ASL to English project..."
echo ""

# Activate venv
source venv/bin/activate

echo "📦 Installing dependencies for signtalk.py (simpler option)..."
cd ASL_to_English
pip install --upgrade pip
pip install opencv-python mediapipe scikit-learn pyttsx3 numpy

echo ""
echo "✅ Basic setup complete!"
echo ""
echo "To test webcam:"
echo "  python webcam_smoke_test.py"
echo ""
echo "To run signtalk:"
echo "  python signtalk.py --labels 'Hello,Thanks,Yes,No,Please,ILoveYou'"
echo ""
