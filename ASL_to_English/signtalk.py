#!/usr/bin/env python3
"""
SignTalk: real-time sign → text/speech (DIY, no big model needed)

Install:
  pip install opencv-python mediapipe scikit-learn pyttsx3 numpy

Run (example labels):
  python signtalk.py --labels "Hello,Thanks,Yes,No"

Controls in the window:
  1..N  - switch active label (based on --labels)
  c     - capture a training sample for the active label (requires a detected hand)
  u     - undo last sample for the active label
  t     - train the classifier on collected samples
  r     - toggle live recognition on/off
  g     - toggle grayscale preview
  n     - cycle to next camera index (0,1,2 by default; change with --indices)
  s     - speak the current stable recognition
  w     - save dataset to file (or auto-saves on exit)
  backspace - clear the live transcript
  q/ESC - quit

How it works:
- Uses MediaPipe Hands to extract 21 3D landmarks from your hand(s).
- Converts landmarks into a normalized, translation & scale-invariant feature vector.
- Trains a simple scikit-learn classifier (KNN by default) on your captured samples.
- During recognition, shows the top prediction and appends it to a transcript when stable.
"""
import argparse, os, time, sys, math, pickle, platform
from collections import defaultdict, deque

import numpy as np
import cv2

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# MediaPipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, default="Hello,Thanks,Yes,No,Please,ILoveYou,This,isSay,Less",
                    help="Comma-separated list of target words/signs")
    ap.add_argument("--indices", type=str, default="0,1,2", help="Comma-separated camera indices to try/cycle")
    ap.add_argument("--hands", type=int, default=1, help="Number of hands to detect (1 or 2)")
    ap.add_argument("--min_detect", type=float, default=0.6, help="MediaPipe min_detection_confidence")
    ap.add_argument("--min_track", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
    ap.add_argument("--algo", type=str, default="knn", choices=["knn"], help="Classifier algorithm")
    ap.add_argument("--neighbors", type=int, default=5, help="K for KNN")
    ap.add_argument("--save", type=str, default="signtalk_data.pkl", help="Path to save/load dataset (pickle). Default: signtalk_data.pkl")
    ap.add_argument("--load", type=str, default="", help="Optional path to load dataset (pickle). If not provided, will try --save file.")
    ap.add_argument("--stability", type=int, default=10, help="Frames required for a stable prediction")
    return ap.parse_args()

def normalize_landmarks(landmarks):
    """Normalize landmarks to be translation/scale invariant.
    - Translate so that wrist (index 0) is at origin.
    - Scale by distance between wrist (0) and middle MCP (9), fallback to global std.
    - Flatten to 63-dim (21 * (x,y,z)).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21,3) normalized by image
    origin = pts[0].copy()
    pts -= origin
    # scale by a reference bone length
    ref = np.linalg.norm(pts[9])  # middle MCP relative to wrist
    if ref < 1e-6:
        ref = np.std(pts) + 1e-6
    pts /= ref
    return pts.flatten()  # (63,)

def draw_hud(frame, text_lines, y0=26):
    for i, t in enumerate(text_lines):
        y = y0 + i*28
        cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def speak(text):
    if not text:
        return
    if TTS_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass
    # macOS fallback
    if sys.platform == "darwin":
        os.system(f"say '{text}'")

def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    return cap


def call_model():
    args = parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    # dataset: dict[label] -> list of feature vectors
    dataset = defaultdict(list)

    # load dataset
    load_file = args.load if args.load else (args.save if os.path.exists(args.save) else None)
    if load_file and os.path.exists(load_file):
        with open(load_file, "rb") as f:
            loaded = pickle.load(f)
            for k, v in loaded.items():
                dataset[k] = v
        print(f"Loaded dataset from {load_file} ({sum(len(v) for v in dataset.values())} samples)")

    clf = None
    gray = False

    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    current = 0
    cap = None
    for j in range(len(indices)):
        cap = open_cam(indices[j])
        if cap is not None:
            current = j
            break
    if cap is None:
        raise SystemExit(f"Could not open any camera from {indices}")

    # mediapipe hands
    hands = mp_hands.Hands(
        max_num_hands=max(1, min(args.hands, 2)),
        min_detection_confidence=args.min_detect,
        min_tracking_confidence=args.min_track)

    # recognition state
    recognizing = False
    last_pred = None
    stable_queue = deque(maxlen=args.stability)
    transcript = []

    print("Controls: 1..N=switch label  c=capture  u=undo  t=train  r=toggle recognize  n=next cam  g=gray  s=speak  backspace=clear  q/ESC=quit")
    active = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            current = (current + 1) % len(indices)
            cap = open_cam(indices[current])
            if cap is None:
                continue
            else:
                continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = hands.process(img_rgb)
        lm = None
        if res.multi_hand_landmarks:
            # take the first detected hand
            lm = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # build HUD
        counts = {lab: len(dataset[lab]) for lab in labels}
        line1 = f"[cam {indices[current]}] Active [{active+1}] {labels[active]} | c=cap u=undo t=train r=recognize n=next g=gray s=speak q=quit"
        line2 = "Samples: " + " | ".join([f"{i+1}:{labels[i]}={counts[labels[i]]}" for i in range(len(labels))])
        line3 = "Transcript: " + " ".join(transcript[-10:])
        # recognition
        pred_text = ""
        if recognizing and clf is not None and lm is not None:
            feat = normalize_landmarks(lm.landmark).reshape(1, -1)
            pred_id = clf.predict(feat)[0]
            proba = None
            if hasattr(clf, "predict_proba"):
                try:
                    proba = np.max(clf.predict_proba(feat))
                except Exception:
                    proba = None
            pred_text = id_to_label[pred_id]
            last_pred = (pred_text, proba if proba is not None else 1.0)
            stable_queue.append(pred_text)
            # if stable over last K frames, append to transcript
            if len(stable_queue) == stable_queue.maxlen and len(set(stable_queue)) == 1:
                transcript.append(pred_text)
                speak(pred_text)
                stable_queue.clear()

        if last_pred:
            lp, conf = last_pred
            line4 = f"Recognizing: {lp} ({conf:.2f})" if conf is not None else f"Recognizing: {lp}"
        else:
            line4 = "Recognizing: (idle)" if recognizing else "Recognizing: OFF"

        if gray:
            view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        else:
            view = frame

        draw_hud(view, [line1, line2, line3, line4])
        cv2.imshow("SignTalk", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('g'):
            gray = not gray
        elif key == ord('n'):
            cap.release()
            tried = 0
            while tried < len(indices):
                current = (current + 1) % len(indices)
                cap = open_cam(indices[current])
                if cap is not None:
                    print(f"Switched to camera index {indices[current]}")
                    break
                tried += 1
            if cap is None:
                raise SystemExit("No working cameras available.")
        elif key == ord('c'):
            if lm is not None:
                feat = normalize_landmarks(lm.landmark)
                dataset[labels[active]].append(feat)
                print(f"[+] captured {labels[active]}  (total {len(dataset[labels[active]])})")
            else:
                print("No hand detected for capture.")
        elif key == ord('u'):
            if dataset[labels[active]]:
                dataset[labels[active]].pop()
                print(f"[-] removed last sample of {labels[active]}  (total {len(dataset[labels[active]])})")
        elif key == ord('t'):
            # build X, y
            X, y = [], []
            for lab, feats in dataset.items():
                X.extend(feats)
                y.extend([label_to_id[lab]] * len(feats))
            if len(set(y)) < 2:
                print("Need samples from at least 2 labels to train.")
            elif len(X) < 10:
                print("Collect more samples (>=10 total recommended).")
            else:
                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.int32)
                clf_local = KNeighborsClassifier(n_neighbors=max(1, min(args.neighbors, len(set(y)))))
                clf_local.fit(X, y)
                clf = clf_local
                # quick resubstitution accuracy (just a sanity check)
                acc = accuracy_score(y, clf.predict(X))
                print(f"[train] done. resub-accuracy ~ {acc:.2f} on {len(y)} samples.")
                if args.save:
                    with open(args.save, "wb") as f:
                        pickle.dump(dataset, f)
                        print(f"[save] dataset -> {args.save}")
        elif key == ord('r'):
            recognizing = not recognizing
            stable_queue.clear()
            print("Recognize:", recognizing)
        elif key == ord('s'):
            if last_pred:
                speak(last_pred[0])
        elif key == ord('w'):  # 'w' for write/save
            if args.save:
                with open(args.save, "wb") as f:
                    pickle.dump(dataset, f)
                    print(f"[save] dataset -> {args.save} ({sum(len(v) for v in dataset.values())} samples)")
            else:
                # Auto-generate filename if --save not provided
                save_file = "signtalk_data.pkl"
                with open(save_file, "wb") as f:
                    pickle.dump(dataset, f)
                    print(f"[save] dataset -> {save_file} ({sum(len(v) for v in dataset.values())} samples)")
        elif key == 8:  # backspace
            transcript.clear()
        elif key in range(ord('1'), ord('1') + len(labels)):
            active = key - ord('1')

    cap.release()
    cv2.destroyAllWindows()
    
    # Auto-save on exit if dataset has data
    if dataset and sum(len(v) for v in dataset.values()) > 0:
        save_file = args.save if args.save else "signtalk_data.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(dataset, f)
            print(f"\n[auto-save] dataset -> {save_file} ({sum(len(v) for v in dataset.values())} samples)")

def train_model(pkl_path="signtalk_data.pkl", neighbors=5):
    """Train model from pickle file. Returns classifier, id_to_label mapping, and hands processor."""
    # Load dataset
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Training data not found: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    # Extract labels from dataset (not from args)
    labels = list(dataset.keys())
    if not labels:
        raise ValueError("Dataset is empty")
    
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    
    # Build X, y
    X, y = [], []
    for lab, feats in dataset.items():
        if len(feats) > 0:
            X.extend(feats)
            y.extend([label_to_id[lab]] * len(feats))
    
    if len(X) == 0:
        raise ValueError("No training samples found")
    
    if len(set(y)) < 2:
        raise ValueError("Need samples from at least 2 labels")
    
    # Train classifier
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    n_neighbors = min(neighbors, len(set(y)))
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    
    return clf, id_to_label, hands

def predict_from_image(image, clf, id_to_label, hands):
    """Predict ASL sign from image. Requires trained classifier and hands processor."""
    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect hand
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None, 0.0, "no_hand_detected"
    
    # Get landmarks and predict
    lm = res.multi_hand_landmarks[0]
    feat = normalize_landmarks(lm.landmark).reshape(1, -1)
    pred_id = clf.predict(feat)[0]
    confidence = clf.predict_proba(feat)[0].max()
    
    return id_to_label[pred_id], confidence, "success"

if __name__ == "__main__":    
    clf, id_to_label, hands = train_model("signtalk_data.pkl")
    print("Finished training model")
    
    #to train model by yourself, simply comment out the top of the main function and uncomment the call_model function
    #call_model()
    