"""
Run the detector on a video file or a live webcam, with a running violation count.

    python predict_video.py --source 0                      # webcam, live window
    python predict_video.py --source site.mp4 --save out.mp4

Press q to quit the live window. On macOS the first webcam run triggers the
camera permission prompt for your terminal.
"""
import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO

BANNER_OK = (34, 197, 94)      # BGR green
BANNER_ALERT = (68, 68, 239)   # BGR red


def find_weights(path):
    if os.path.exists(path):
        return path
    candidates = sorted(glob.glob("runs/detect/*/weights/best.pt"), key=os.path.getmtime)
    if not candidates:
        raise SystemExit("No trained weights found — run: python train.py")
    return candidates[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="results/best.pt")
    p.add_argument("--source", default="0", help="webcam index (0) or a path to a video file")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", default=None, help="write an annotated mp4 here")
    p.add_argument("--show", action="store_true", default=None, help="force the preview window")
    args = p.parse_args()

    model = YOLO(find_weights(args.weights))
    head_id = next(i for i, n in model.names.items() if n == "head")
    source = int(args.source) if args.source.isdigit() else args.source
    show = args.show if args.show is not None else isinstance(source, int)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames, alert_frames, started = 0, 0, time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        annotated = result.plot(line_width=2)
        classes = result.boxes.cls.tolist()
        heads = sum(c == head_id for c in classes)
        total = len(classes)
        frames += 1
        alert_frames += heads > 0

        label = (f"VIOLATION: {heads}/{total} heads unprotected" if heads
                 else f"OK: {total} heads, all protected" if total else "no heads detected")
        colour = BANNER_ALERT if heads else BANNER_OK
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 38), colour, -1)
        cv2.putText(annotated, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Hard hat detection - q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer:
        writer.release()
        print(f"Wrote {args.save}")
    cv2.destroyAllWindows()
    elapsed = time.time() - started
    print(f"{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS), "
          f"{alert_frames} frames with at least one violation "
          f"({alert_frames / max(frames, 1):.0%})")


if __name__ == "__main__":
    main()
