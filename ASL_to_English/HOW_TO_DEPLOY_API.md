# How to Deploy Your Model as an API - Step by Step Guide

This guide teaches you the general approach to deploying any ML model as an API.

## Understanding the Components

### 1. What is an API?
An API (Application Programming Interface) lets other programs call your model over the network. Instead of running Python scripts, you send HTTP requests.

### 2. What You Need
- **Model**: Your trained model (signtalk_data.pkl)
- **Web Framework**: Flask or FastAPI (handles HTTP requests)
- **Server**: Runs your API (can be local or cloud)

## Step-by-Step Process

### Step 1: Choose a Web Framework

**Option A: Flask** (simpler, more common)
```bash
pip install flask
```

**Option B: FastAPI** (modern, auto-docs, better for APIs)
```bash
pip install fastapi uvicorn
```

**Recommendation**: Use FastAPI - it's designed for APIs and generates automatic documentation.

### Step 2: Understand Your Model's Input/Output

For signtalk.py:
- **Input**: Image file (JPG/PNG) with a hand in it
- **Processing**: 
  1. Load image
  2. Detect hand with MediaPipe
  3. Extract landmarks
  4. Normalize landmarks
  5. Run through classifier
- **Output**: Prediction (string) + confidence (float)

### Step 3: Create the API Structure

Basic FastAPI structure:
```python
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read uploaded image
    # 2. Process with your model
    # 3. Return prediction
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4: Load Your Model

```python
import pickle

# Load at startup (once, not per request)
with open("signtalk_data.pkl", "rb") as f:
    dataset = pickle.load(f)

# Train classifier (or load pre-trained)
# ... your training code ...
```

**Key Point**: Load model ONCE when server starts, not on every request.

### Step 5: Handle Image Upload

```python
from fastapi import File, UploadFile
import cv2
import numpy as np

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Now process with your model...
```

### Step 6: Process with Your Model

Extract the prediction logic from signtalk.py:
```python
# From signtalk.py - normalize_landmarks function
def normalize_landmarks(landmarks):
    # ... copy from signtalk.py ...

# Use MediaPipe to detect hand
results = hands.process(image_rgb)
hand_landmarks = results.multi_hand_landmarks[0]

# Normalize and predict
features = normalize_landmarks(hand_landmarks.landmark)
prediction = classifier.predict([features])[0]
confidence = classifier.predict_proba([features])[0].max()
```

### Step 7: Return JSON Response

```python
return {
    "prediction": "Hello",
    "confidence": 0.95,
    "status": "success"
}
```

### Step 8: Test Your API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

**Using Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
print(response.json())
```

## Common Patterns

### Pattern 1: Health Check Endpoint
Always include this:
```python
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": classifier is not None}
```

### Pattern 2: Error Handling
```python
try:
    # your code
    return {"prediction": result}
except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
```

### Pattern 3: CORS (for web frontends)
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Deployment Options

### Local Testing
```bash
python api_server.py
# Runs on http://localhost:8000
```

### Production Options

**1. Gunicorn (for production)**
```bash
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

**2. Docker**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]
```

**3. Cloud Platforms**
- **Heroku**: Push to git, auto-deploys
- **AWS Lambda**: Serverless (needs adaptation)
- **Google Cloud Run**: Container-based
- **Railway/Render**: Simple git-based deployment

## Key Concepts to Remember

1. **Model Loading**: Load once at startup, reuse for all requests
2. **Async/Await**: Use async functions for I/O operations (file uploads)
3. **Error Handling**: Always handle missing files, bad images, no hand detected
4. **Response Format**: Return consistent JSON structure
5. **Documentation**: FastAPI auto-generates docs at `/docs`

## Your Task: Build It Step by Step

1. **Start simple**: Create a basic FastAPI app that returns "Hello World"
2. **Add health check**: Create `/health` endpoint
3. **Add file upload**: Create `/predict` that accepts an image (don't process yet)
4. **Add MediaPipe**: Detect hand in uploaded image
5. **Add model**: Load your trained model
6. **Add prediction**: Run prediction and return result
7. **Add error handling**: Handle edge cases
8. **Test**: Use curl or Python requests to test

## Debugging Tips

- Use `print()` statements to see what's happening
- Check FastAPI auto-docs at `http://localhost:8000/docs`
- Test each step independently
- Use `curl -v` for verbose HTTP output

## Next Steps After Basic API Works

1. Add authentication (API keys)
2. Add rate limiting
3. Add logging
4. Add metrics/monitoring
5. Optimize for production (caching, async processing)

