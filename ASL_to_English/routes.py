from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request
import json
import numpy as np
import cv2
import ASL_to_English.signtalk as signtalk

router = APIRouter()

#health check endpoint
@router.get("/health")
async def health():
    return {"status": "ok"}

#predict endpoint that accepts an image and returns a prediction
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

    #Return prediction
    return {
        "prediction": prediction,
        "confidence": float(confidence) if confidence else 0.0,
        "status": status
    }