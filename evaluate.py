"""
Evaluate a trained detector on the held-out test split and write the report assets.

    python evaluate.py --weights runs/detect/yolov8s_640px_40e/weights/best.pt

Produces, in results/:
    best.pt                       the trained weights the demo app loads
    metrics.json                  every number quoted in the README
    metrics.md                    the same numbers as a markdown table
    predictions_grid.png          model output on 12 test images
    ground_truth_vs_prediction.png  side-by-side on 4 test images
    confusion_matrix.png, pr_curve.png, results.png   copied from the run directory

The test split is touched exactly here. Model selection (which epoch to keep) is
done by Ultralytics on the val split, so the test numbers stay honest.
"""
import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from ultralytics import YOLO

RESULTS = "results"
COLORS = {"helmet": "#22c55e", "head": "#ef4444", "person": "#3b82f6"}


def load_split(data_yaml, split):
    with open(data_yaml) as fh:
        cfg = yaml.safe_load(fh)
    root = cfg["path"]
    image_dir = os.path.join(root, cfg[split])
    return sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png"))


def ground_truth_boxes(image_path):
    """Read the original VOC annotation for one image: [(label, xmin, ymin, xmax, ymax)]."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    xml_path = os.path.join("data", "annotations", f"{stem}.xml")
    boxes = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        b = obj.find("bndbox")
        boxes.append((
            obj.find("name").text,
            *(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")),
        ))
    return boxes


def n_heads(image_path):
    return sum(label == "head" for label, *_ in ground_truth_boxes(image_path))


def prediction_grid(model, images, dest, conf, cols=4):
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, image_path in zip(axes.ravel(), images):
        result = model.predict(image_path, conf=conf, verbose=False)[0]
        ax.imshow(result.plot(line_width=2, font_size=9)[:, :, ::-1])  # BGR -> RGB
        counts = {}
        for cls in result.boxes.cls.tolist():
            name = model.names[int(cls)]
            counts[name] = counts.get(name, 0) + 1
        violations = counts.get("head", 0)
        title = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nothing detected"
        ax.set_title(title, fontsize=10, color="#ef4444" if violations else "#16a34a")
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.suptitle(f"Predictions on held-out test images (conf >= {conf})", fontsize=13)
    fig.tight_layout()
    fig.savefig(dest, dpi=130, bbox_inches="tight")
    plt.close(fig)


def gt_vs_pred(model, images, dest, conf):
    fig, axes = plt.subplots(len(images), 2, figsize=(9, 4.6 * len(images)))
    for row, image_path in zip(axes, images):
        image = Image.open(image_path)
        row[0].imshow(image)
        for label, xmin, ymin, xmax, ymax in ground_truth_boxes(image_path):
            row[0].add_patch(patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, lw=2, edgecolor=COLORS.get(label, "#eab308")))
        row[0].set_title("ground truth (Pascal VOC)", fontsize=11)
        result = model.predict(image_path, conf=conf, verbose=False)[0]
        row[1].imshow(result.plot(line_width=2, font_size=9)[:, :, ::-1])
        row[1].set_title(f"prediction (conf >= {conf})", fontsize=11)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for the figures")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    project = os.path.abspath("runs/detect")  # see the note in train.py
    model = YOLO(args.weights)
    run_dir = os.path.dirname(os.path.dirname(args.weights))

    metrics = {}
    for split in ("val", "test"):
        m = model.val(data=args.data, split=split, imgsz=args.imgsz, device=args.device,
                      project=project, name=f"{os.path.basename(run_dir)}_{split}",
                      exist_ok=True, plots=True, verbose=False)
        metrics[split] = {
            "overall": {"precision": m.box.mp, "recall": m.box.mr,
                        "mAP50": m.box.map50, "mAP50_95": m.box.map},
            "per_class": {
                model.names[c]: {"precision": m.box.p[i], "recall": m.box.r[i],
                                 "mAP50": m.box.ap50[i], "mAP50_95": m.box.ap[i]}
                for i, c in enumerate(m.box.ap_class_index)
            },
            "speed_ms": m.speed,
        }
        o = metrics[split]["overall"]
        print(f"{split:<5} mAP50 {o['mAP50']:.4f}  mAP50-95 {o['mAP50_95']:.4f}  "
              f"P {o['precision']:.4f}  R {o['recall']:.4f}")

    test_images = load_split(args.data, "test")
    metrics["dataset"] = {
        split: len(load_split(args.data, split)) for split in ("train", "val", "test")
    }
    metrics["model"] = {"weights": args.weights, "imgsz": args.imgsz,
                        "classes": list(model.names.values()),
                        "parameters": sum(p.numel() for p in model.model.parameters())}

    with open(os.path.join(RESULTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    with open(os.path.join(RESULTS, "metrics.md"), "w") as fh:
        for split in ("val", "test"):
            fh.write(f"\n### {split} split\n\n")
            fh.write("| class | precision | recall | mAP@50 | mAP@50-95 |\n")
            fh.write("|---|---|---|---|---|\n")
            for name, row in metrics[split]["per_class"].items():
                fh.write(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | "
                         f"{row['mAP50']:.3f} | {row['mAP50_95']:.3f} |\n")
            o = metrics[split]["overall"]
            fh.write(f"| **all** | **{o['precision']:.3f}** | **{o['recall']:.3f}** | "
                     f"**{o['mAP50']:.3f}** | **{o['mAP50_95']:.3f}** |\n")

    # Figures: half the picks are images with a real violation, so the gallery
    # shows the case the model exists for rather than 12 compliant crews.
    rng = random.Random(args.seed)
    with_head = [i for i in test_images if n_heads(i) > 0]
    without = [i for i in test_images if n_heads(i) == 0]
    picks = rng.sample(with_head, 8) + rng.sample(without, 4)
    prediction_grid(model, picks, os.path.join(RESULTS, "predictions_grid.png"), args.conf)
    gt_vs_pred(model, rng.sample(with_head, 3) + rng.sample(without, 1),
               os.path.join(RESULTS, "ground_truth_vs_prediction.png"), args.conf)

    # The demo app loads results/best.pt, and the repo ships it so the project is
    # usable without retraining; the example images give the app a gallery.
    shutil.copy2(args.weights, os.path.join(RESULTS, "best.pt"))
    example_dir = os.path.join(RESULTS, "examples")
    os.makedirs(example_dir, exist_ok=True)
    for image_path in picks[:6]:
        shutil.copy2(image_path, os.path.join(example_dir, os.path.basename(image_path)))

    for src, dst in [
        (f"{run_dir}/results.png", "training_curves.png"),
        (f"{run_dir}_test/confusion_matrix_normalized.png", "confusion_matrix.png"),
        (f"{run_dir}_test/BoxPR_curve.png", "pr_curve.png"),
        (f"{run_dir}_test/BoxF1_curve.png", "f1_curve.png"),
        (f"{run_dir}/labels.jpg", "label_distribution.jpg"),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(RESULTS, dst))

    print(f"\nWrote {RESULTS}/ — metrics.json, metrics.md and the figures.")


if __name__ == "__main__":
    main()
