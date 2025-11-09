# ASL to Text Translator

Frontend repo that connects to this backend server: https://github.com/bradleyyang/hacktrent-app

This repository provides a complete system for translating **American Sign Language (ASL)** video frames into **English text**, with an optional text-to-speech output layer.  
It includes a trained gesture recognition model, a FastAPI backend for inference, and a fully containerized runtime for easy deployment.

The general approach was originally inspired by open-source ASL recognition research, but **the dataset preparation, model training workflow, backend API architecture, Docker deployment setup, and integration design in this project are original.**

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| **Custom-trained ASL recognition model** | Converts ASL hand gestures to English tokens. |
| **REST API (FastAPI)** | `/predict` for translation, `/health` for status checking. |
| **Dockerized Deployment** | Run locally or on cloud providers like Render, Railway, DigitalOcean, etc. |
| **Optional Text-to-Speech** | Uses ElevenLabs to convert recognized text into natural-sounding speech. |

---

## 🧱 Project Structure
```
ASL_to_text/
├─ ASL_to_English/ # Packaged model assets & notebooks
│ ├─ annotations/ # Label data / TF records (if present)
│ ├─ my_ssd_mobnet/ # Model pipeline / checkpoints / label_map
│ ├─ test/ # Test images / clips
│ ├─ Signlangtranslator.ipynb # Training / experimentation notebook
│ └─ realtime_image_collection.ipynb # Webcam image collection notebook
├─ server.py # FastAPI app 
├─ requirements.txt # original python dep for training model (not needed unless you want to retrain model yourself)
├─ Dockerfile # Production container
├─ docker-compose.yml # Local orchestration (optional)
├─ GET_STARTED.sh # Convenience bootstrap script
├─ CLEAN_SETUP.sh # Clean env / reinstall helper
├─ .gitignore
├─ .dockerignore
└─ README.md # (this file)
```

## 🚀 Run Locally

### 1. Create virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ASL_to_English/requirements_api.txt
pip install -r ASL_to_English/requirements_signtalk.txt
```

### 2. Start the API server
```bash
python server.py
```

### 3. Test prediction endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/your/image.jpg"
```

---

## 🧭 Acknowledgements
The model trained is forked from this repo:
https://github.com/priiyaanjaalii0611/ASL_to_English

However, the training process, API backend, system architecture, and deployment workflow in this repository are independently developed.

---

## 🐳 Run with Docker

### Build the image
```bash
docker build -t asl-recognition-api .
```

### Run the container
```bash
docker run -p 8000:8000 \
  -e ELEVENLABS_API_KEY="your_key_here" \
  -e GEMINI_API_KEY="your_key_here" \
  asl-recognition-api
```

API will be available at:
- http://localhost:8000

---

## 🎤 Optional Speech Output
If you want the recognized text to be spoken aloud:
```bash
export ELEVENLABS_API_KEY="your_key_here"
```

The `/predict` endpoint automatically returns audio along with the prediction.

---

## 🗺️ Future Improvements
- Webcam real-time streaming translation
- Larger vocabulary training

