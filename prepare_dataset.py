"""
Convert the Pascal VOC annotations in data/ to YOLO format and build the splits.

Reads          data/images/*.png  +  data/annotations/*.xml   (what get_data.py wrote)
Writes         data/yolo/images/{train,val,test}/*.png
               data/yolo/labels/{train,val,test}/*.txt
               data.yaml

Three decisions are baked in here, all of them explained in the README:

1. Two classes, not three. The raw dataset has `helmet` (18,966 boxes),
   `head` (5,785) and `person` (751). `person` marks a whole body rather than a
   head, so it answers a different question than "is this worker protected?" and
   is 40x rarer than `helmet`. We keep helmet/head — the safety-relevant pair —
   and drop `person`. Set KEEP_PERSON = True to train the 3-class variant.

2. A stratified split. Only ~28% of images contain a bare `head`, and those are
   the images the safety use case is about. A plain random split can leave the
   val/test sets with noticeably different violation rates than train, which
   makes the metrics harder to trust, so images are split 70/20/10 *within*
   each group (has-a-bare-head vs. all-helmets).

3. Hard links, not copies. The YOLO tree points at the same image bytes on disk,
   so the split costs ~0 MB instead of another 250 MB.

Usage:

    python prepare_dataset.py
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

IMAGES = os.path.join("data", "images")
ANNOTATIONS = os.path.join("data", "annotations")
YOLO_ROOT = os.path.join("data", "yolo")
DATA_YAML = "data.yaml"

KEEP_PERSON = False
CLASSES = ["helmet", "head"] + (["person"] if KEEP_PERSON else [])
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42


def voc_to_yolo(xml_path):
    """One VOC file -> a list of '<class> <cx> <cy> <w> <h>' lines, all normalised."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w, img_h = int(size.find("width").text), int(size.find("height").text)

    lines, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        labels.append(name)
        if name not in CLASS_ID:
            continue
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))

        # VOC corners -> YOLO centre/size, normalised, clamped to the image.
        xmin, xmax = max(0.0, min(xmin, xmax)), min(float(img_w), max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(float(img_h), max(ymin, ymax))
        w, h = (xmax - xmin) / img_w, (ymax - ymin) / img_h
        cx, cy = (xmin + xmax) / 2 / img_w, (ymin + ymax) / 2 / img_h
        if w <= 0 or h <= 0:  # degenerate box, nothing to learn from
            continue
        lines.append(f"{CLASS_ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, labels


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:  # different filesystem, or a filesystem without hard links
        shutil.copy2(src, dst)


def split_images(stems, has_head):
    """70/20/10, stratified on whether the image contains an unprotected head."""
    rng = random.Random(SEED)
    assignment = {}
    for group in (True, False):
        members = sorted(s for s in stems if has_head[s] is group)
        rng.shuffle(members)
        n = len(members)
        n_train = round(n * SPLITS["train"])
        n_val = round(n * SPLITS["val"])
        for split, chunk in (
            ("train", members[:n_train]),
            ("val", members[n_train:n_train + n_val]),
            ("test", members[n_train + n_val:]),
        ):
            for stem in chunk:
                assignment[stem] = split
    return assignment


def main():
    stems = sorted(f[:-4] for f in os.listdir(ANNOTATIONS) if f.endswith(".xml"))
    assert stems, f"no annotations in {ANNOTATIONS}/ — run get_data.py first"

    print(f"Converting {len(stems):,} Pascal VOC files to YOLO format...")
    yolo_labels, has_head, raw_counts = {}, {}, Counter()
    for stem in stems:
        lines, labels = voc_to_yolo(os.path.join(ANNOTATIONS, f"{stem}.xml"))
        yolo_labels[stem] = lines
        raw_counts.update(labels)
        has_head[stem] = "head" in labels

    dropped = sum(n for name, n in raw_counts.items() if name not in CLASS_ID)
    print(f"  raw boxes: " + ", ".join(f"{k} {v:,}" for k, v in raw_counts.most_common()))
    print(f"  kept classes {CLASSES}" + (f", dropped {dropped:,} boxes" if dropped else ""))
    print(f"  images with at least one bare head: {sum(has_head.values()):,}"
          f" ({sum(has_head.values()) / len(stems):.1%})")

    if os.path.isdir(YOLO_ROOT):
        shutil.rmtree(YOLO_ROOT)
    for split in SPLITS:
        os.makedirs(os.path.join(YOLO_ROOT, "images", split))
        os.makedirs(os.path.join(YOLO_ROOT, "labels", split))

    assignment = split_images(stems, has_head)
    per_split = {split: Counter() for split in SPLITS}
    n_images = Counter()
    empty = 0
    for stem, split in assignment.items():
        link(os.path.join(IMAGES, f"{stem}.png"),
             os.path.join(YOLO_ROOT, "images", split, f"{stem}.png"))
        lines = yolo_labels[stem]
        with open(os.path.join(YOLO_ROOT, "labels", split, f"{stem}.txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        n_images[split] += 1
        empty += not lines
        for line in lines:
            per_split[split][CLASSES[int(line.split()[0])]] += 1

    print(f"\n{'split':<8}{'images':>8}" + "".join(f"{c:>10}" for c in CLASSES) + f"{'head rate':>12}")
    for split in SPLITS:
        counts = per_split[split]
        head_rate = counts["head"] / max(sum(counts.values()), 1)
        print(f"{split:<8}{n_images[split]:>8,}"
              + "".join(f"{counts[c]:>10,}" for c in CLASSES)
              + f"{head_rate:>11.1%}")
    total = sum(per_split[s][c] for s in SPLITS for c in CLASSES)
    print(f"{'total':<8}{sum(n_images.values()):>8,}"
          + "".join(f"{sum(per_split[s][c] for s in SPLITS):>10,}" for c in CLASSES))
    if empty:
        print(f"\n{empty} images have no boxes of the kept classes — YOLO treats them as "
              f"background examples, which is useful, so they stay in.")

    with open(DATA_YAML, "w") as fh:
        fh.write(
            "# Generated by prepare_dataset.py — regenerate rather than edit by hand.\n"
            f"path: {os.path.abspath(YOLO_ROOT)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
        )
    print(f"\nWrote {DATA_YAML}")
    print("Now run: python train.py")


if __name__ == "__main__":
    main()
