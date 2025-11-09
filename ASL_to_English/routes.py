from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request
import json
import base64
import numpy as np
import cv2
import ASL_to_English.signtalk as signtalk
import ASL_to_English.api_calls as api_calls

router = APIRouter()

#health check endpoint
@router.get("/health")
async def health():
    return {"status": "ok"}

#predict endpoint that accepts an image and returns audio + prediction
@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    #Access app state through request
    if not hasattr(request.app.state, 'classifier') or request.app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server may still be initializing.")
    
    #Read uploaded image
    contents = await file.read()    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please ensure it's a valid image file.")

    #Process with model
    prediction, confidence, status = signtalk.predict_from_image(
        image, 
        request.app.state.classifier, 
        request.app.state.id_to_label, 
        request.app.state.hands
    )

    #Get audio
    audio = api_calls.get_audio(prediction)

    if audio is None:
        raise HTTPException(status_code=500, detail="Error getting audio.")

    # Encode audio as base64 for JSON response
    chunks = list(audio)                 # or: b"".join(audio) directly
    audio_bytes = b"".join(chunks)       # bytes now
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    #Return prediction
    return {
        "audio": audio_base64,
        "prediction": prediction,
        "confidence": float(confidence) if confidence else 0.0,
        "status": status
    }