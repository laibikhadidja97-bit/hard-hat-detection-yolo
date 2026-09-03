"""
Train a YOLO detector for hard-hat compliance.

    python train.py                                  # YOLOv8s, 30 epochs, 416 px
    python train.py --imgsz 640 --batch 8            # higher resolution, ~3x slower
    python train.py --device cpu                     # if MPS/CUDA misbehaves

Equivalent to the Ultralytics CLI form:

    yolo detect train data=data.yaml model=yolov8s.pt epochs=30 imgsz=416 batch=16

but wrapped in a script, so the exact configuration behind the reported numbers
lives in the repo. Results land in runs/detect/<name>/, with the weights in
weights/best.pt and every training curve written alongside them.

416 px is the images' native size. 640 px (Ultralytics' default) was measured first
and abandoned: on a 16 GB M3 it pushed unified memory past 7 GB, the machine started
swapping, and one iteration went from 1.0 s to ~50 s. The README records the numbers.
"""
import argparse
import os
import time

import torch
from ultralytics import YOLO


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():  # Apple silicon
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint to fine-tune")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=416,
                   help="the images are natively 416x416; see the README on why 640 was dropped")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience, in epochs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    p.add_argument("--name", default=None, help="run directory name (default: derived from the model)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = pick_device(args.device)
    # Ultralytics resolves a relative `project` against its own global runs_dir
    # (~/Library/.../Ultralytics/settings.json), which would scatter runs outside
    # this repo. Pin it to the project folder instead.
    project = os.path.abspath("runs/detect")
    name = args.name or f"{args.model.split('.')[0]}_{args.imgsz}px_{args.epochs}e"
    print(f"Training {args.model} on {device} -> runs/detect/{name}")

    model = YOLO(args.model)
    started = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        val=True,
    )
    minutes = (time.time() - started) / 60
    print(f"\nTraining finished in {minutes:.1f} min")

    # Ultralytics validates on the val split each epoch; report the final numbers
    # for the record, then leave the held-out test split to evaluate.py.
    metrics = model.val(data=args.data, split="val", imgsz=args.imgsz, device=device,
                        project=project, name=f"{name}_val", exist_ok=True)
    print(f"val  mAP50 {metrics.box.map50:.4f}   mAP50-95 {metrics.box.map:.4f}   "
          f"P {metrics.box.mp:.4f}   R {metrics.box.mr:.4f}")
    print(f"\nWeights: runs/detect/{name}/weights/best.pt")
    print("Now run: python evaluate.py --weights runs/detect/%s/weights/best.pt" % name)


if __name__ == "__main__":
    main()
