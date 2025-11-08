#!/usr/bin/env python3
import argparse, os, time, cv2

def main():
    ap = argparse.ArgumentParser(description="Webcam Smoke Test with camera cycling (n=next)")
    ap.add_argument("--indices", type=str, default="0,1,2", help="Comma-separated camera indices to try/cycle")
    ap.add_argument("--cam", type=int, default=None, help="(Deprecated) Start camera index; use --indices instead")
    args = ap.parse_args()

    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    if args.cam is not None and args.cam not in indices:
        indices = [args.cam] + [i for i in indices if i != args.cam]

    def open_cam(i):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            return None
        return cap

    current = 0
    cap = None
    # find a working one to start
    for j in range(len(indices)):
        cap = open_cam(indices[j])
        if cap is not None:
            current = j
            break
    if cap is None:
        raise SystemExit(f"Could not open any camera from {indices}")

    gray = False
    print("Press g=grayscale, s=save, n=next camera, q/ESC=quit.")
    os.makedirs("./smoke_test_out", exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            # try switching if current died
            cap.release()
            current = (current + 1) % len(indices)
            cap = open_cam(indices[current])
            if cap is None:
                continue
            else:
                continue

        view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray else frame
        if gray:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        cv2.putText(view, "g=grayscale  s=save  n=next cam  q/ESC=quit", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(view, f"cam index: {indices[current]}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam Smoke Test", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('g'):
            gray = not gray
        elif key == ord('s'):
            fn = os.path.join("./smoke_test_out", f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(fn, frame)
            print(f"Saved {fn}")
        elif key == ord('n'):
            cap.release()
            current = (current + 1) % len(indices)
            cap = open_cam(indices[current])
            if cap is None:
                print(f"Could not open camera {indices[current]}")
                # keep trying forward until we get one
                tried = 0
                while tried < len(indices):
                    current = (current + 1) % len(indices)
                    cap = open_cam(indices[current])
                    if cap is not None:
                        break
                    tried += 1
                if cap is None:
                    raise SystemExit("No working cameras available.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()