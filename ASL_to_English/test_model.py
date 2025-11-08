#!/usr/bin/env python3
"""
Quick test script for your trained model
Usage: python test_model.py <image_path>
"""

import sys
import pickle
import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

def normalize_landmarks(landmarks):
    """Normalize landmarks (same as signtalk.py)"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    origin = pts[0].copy()
    pts -= origin
    ref = np.linalg.norm(pts[9])
    if ref < 1e-6:
        ref = np.std(pts) + 1e-6
    pts /= ref
    return pts.flatten()

def load_and_train_model(pkl_path, labels):
    """Load dataset and train classifier"""
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    # Build training data
    X, y = [], []
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    
    for lab in labels:
        if lab in dataset and len(dataset[lab]) > 0:
            X.extend(dataset[lab])
            y.extend([label_to_id[lab]] * len(dataset[lab]))
    
    if len(X) == 0:
        raise ValueError("No training data found!")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    classifier = KNeighborsClassifier(n_neighbors=min(5, len(set(y))))
    classifier.fit(X, y)
    
    print(f"✅ Model loaded: {len(X)} samples, {len(set(y))} classes")
    return classifier, id_to_label

def test_image(image_path, classifier, id_to_label):
    """Test a single image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hand
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        print("❌ No hand detected in image")
        return
    
    # Get landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    features = normalize_landmarks(hand_landmarks.landmark)
    
    # Predict
    features_2d = features.reshape(1, -1)
    prediction_id = classifier.predict(features_2d)[0]
    probabilities = classifier.predict_proba(features_2d)[0]
    
    predicted_label = id_to_label[prediction_id]
    confidence = float(probabilities[prediction_id])
    
    print(f"\n✅ Prediction: {predicted_label}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"\nAll probabilities:")
    for i, prob in enumerate(probabilities):
        label = id_to_label[i]
        print(f"   {label}: {prob:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path> [labels]")
        print("Example: python test_model.py test.jpg 'Hello,Thanks,Yes,No'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else ["Hello", "Thanks", "Yes", "No"]
    labels = [s.strip() for s in labels]
    
    pkl_path = "signtalk_data.pkl"
    
    try:
        classifier, id_to_label = load_and_train_model(pkl_path, labels)
        test_image(image_path, classifier, id_to_label)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

