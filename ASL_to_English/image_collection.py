#!/usr/bin/env python3
"""
Image Collection Tool for ASL-to-English dataset creation.

Keys:
  1..N = switch label (order of --labels)
  SPACE = save frame to active label
  v = toggle burst mode
  n = next camera index
  q/ESC = quit
"""
import argparse, os, time, cv2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data", help="Output directory for collected images")
    ap.add_argument("--labels", type=str, default="Hello,I Love You,Thank you,Please,Yes,No", help="Comma-separated list of labels")
    ap.add_argument("--indices", type=str, default="0,1,2", help="Comma-separated camera indices to try/cycle")
    ap.add_argument("--max", type=int, default=0, help="Optional limit per label (0 = unlimited)")
    return ap.parse_args()

def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    return cap

def draw_hud(frame, active_label, labels, counts, cam_idx):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    import numpy as np
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    txt1 = f"[cam {cam_idx}] Active: [{active_label+1}] {labels[active_label]} | SPACE=save | 1..{len(labels)}=switch | v=burst | n=next cam | q=quit"
    cv2.putText(frame, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    txt2 = "Counts: " + " | ".join([f"{i+1}:{labels[i]}={counts.get(labels[i],0)}" for i in range(len(labels))])
    cv2.putText(frame, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return frame

def main():
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    for label in labels:
        os.makedirs(os.path.join(args.out, label), exist_ok=True)

    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    # preload counts
    counts = {label: len([f for f in os.listdir(os.path.join(args.out, label)) if f.lower().endswith(('.jpg','.jpeg','.png'))]) for label in labels}

    # open first working camera
    current = 0
    cap = None
    for j in range(len(indices)):
        cap = open_cam(indices[j])
        if cap is not None:
            current = j
            break
    if cap is None:
        raise SystemExit(f"Could not open any camera from {indices}")

    active = 0
    save_burst = False
    last_save = 0.0

    print("Controls: SPACE=save | 1..N=switch label | v=burst | n=next cam | q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            # switch on failure
            cap.release()
            current = (current + 1) % len(indices)
            cap = open_cam(indices[current])
            if cap is None:
                continue
            else:
                continue

        frame_vis = draw_hud(frame.copy(), active, labels, counts, indices[current])
        cv2.imshow("Image Collection", frame_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('v'):
            save_burst = not save_burst
        elif key == 32:  # SPACE
            label = labels[active]
            if args.max and counts[label] >= args.max:
                print(f"Reached max {args.max} for label '{label}'")
            else:
                fn = os.path.join(args.out, label, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(fn, frame)
                counts[label] += 1
                print(f"Saved {fn}")
        elif key in range(ord('1'), ord('1') + len(labels)):
            active = key - ord('1')
        elif key == ord('n'):
            cap.release()
            # cycle to the next camera
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

        # Burst mode ~10 FPS
        now = time.time()
        if save_burst and (now - last_save) >= 0.1:
            label = labels[active]
            if not args.max or counts[label] < args.max:
                fn = os.path.join(args.out, label, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(fn, frame)
                counts[label] += 1
                last_save = now
                print(f"[burst] Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()