from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier

import json

router = APIRouter()

class Note(BaseModel):
    id: int
    title: str
    content: str

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read uploaded image
    # 2. Process with your model
    # 3. Return prediction
    pass

@router.on_event("startup")
async def startup_event():
    # Load dataset from pickle file
    with open("signtalk_data.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    # Train classifier once
    # Build X, y from dataset
    X, y = [], []
    for lab, feats in dataset.items():
        X.extend(feats)
        y.extend([label_to_id[lab]] * len(feats))
    
    # Train KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X, y)
    
    print("Model trained and ready!")