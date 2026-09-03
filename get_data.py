"""
Fetch the Hard Hat Detection dataset (5,000 annotated images) into data/

The canonical home of this dataset is Kaggle (`andrewmvd/hard-hat-detection`),
which requires an account and API credentials. The identical images and boxes
are mirrored on the Hugging Face Hub as `Voxel51/hard-hat-detection` (CC0-1.0,
same 5,000 `hard_hat_workers*.png` files), which needs no authentication, so
that is what we pull by default.

The mirror stores the boxes in FiftyOne's JSON format. Every field Pascal VOC
carries is preserved there (size, pose, truncated, occluded, difficult), so this
script writes the annotations back out as VOC XML — the exact layout Kaggle
ships:

    data/images/hard_hat_workers0.png ... hard_hat_workers4999.png
    data/annotations/hard_hat_workers0.xml ... hard_hat_workers4999.xml

That matters: the notebook's VOC -> YOLO converter then reads real XML, and it
works unchanged if you download from Kaggle instead:

    kaggle datasets download -d andrewmvd/hard-hat-detection -p data --unzip

Usage:

    python get_data.py

Downloads ~1.3 GB over 5,000 small files; takes a few minutes on a good
connection and is safe to re-run (it only fetches what is missing).
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

REPO = "https://huggingface.co/datasets/Voxel51/hard-hat-detection/resolve/main/"
OUT_DIR = "data"
IMAGES = os.path.join(OUT_DIR, "images")
ANNOTATIONS = os.path.join(OUT_DIR, "annotations")
SAMPLES = os.path.join(OUT_DIR, "_mirror", "samples.json")

EXPECTED_IMAGES = 5000
EXPECTED_OBJECTS = 25_502  # 18,966 helmet + 5,785 head + 751 person
THREADS = 8  # the Hub answers 429 if you pull much harder than this


def fetch(url, dest, retries=6):
    """Download one file, atomically. Returns False if it was already there.

    The Hub rate-limits bursts with HTTP 429, so failures back off exponentially
    (with jitter) instead of giving up — a 5,000-file pull hits the limiter.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    tmp = f"{dest}.part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.rename(tmp, dest)  # only becomes the real file once fully written
            return True
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt + random.random())
    return True


def download_samples():
    os.makedirs(os.path.dirname(SAMPLES), exist_ok=True)
    if fetch(REPO + "samples.json", SAMPLES):
        print("Downloaded samples.json (8.5 MB of box annotations)")
    with open(SAMPLES) as fh:
        return json.load(fh)["samples"]


def download_images(samples):
    os.makedirs(IMAGES, exist_ok=True)
    missing = [
        (REPO + s["filepath"], os.path.join(IMAGES, os.path.basename(s["filepath"])))
        for s in samples
        if not os.path.exists(os.path.join(IMAGES, os.path.basename(s["filepath"])))
    ]
    if not missing:
        print(f"All {len(samples)} images already present in {IMAGES}/")
        return

    print(f"Downloading {len(missing)} images (~1.3 GB) with {THREADS} threads...")
    done = 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for _ in pool.map(lambda job: fetch(*job), missing):
            done += 1
            if done % 25 == 0 or done == len(missing):
                sys.stdout.write(f"\r  {done}/{len(missing)}  {done / len(missing):6.1%}")
                sys.stdout.flush()
    print()


def to_voc(sample):
    """Rebuild one Pascal VOC XML file from a mirrored FiftyOne sample.

    FiftyOne stores boxes as [x, y, w, h] normalised to the image size, with the
    origin at the top-left corner; VOC stores absolute pixel corners.
    """
    name = os.path.basename(sample["filepath"])
    meta = sample["metadata"]
    width, height = meta["width"], meta["height"]

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(meta.get("num_channels", 3))
    ET.SubElement(root, "segmented").text = "0"

    n_objects = 0
    for det in sample["ground_truth"]["detections"]:
        x, y, w, h = det["bounding_box"]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = det["label"]
        ET.SubElement(obj, "pose").text = det.get("pose", "Unspecified")
        ET.SubElement(obj, "truncated").text = str(int(det.get("truncated", 0)))
        ET.SubElement(obj, "occluded").text = str(int(det.get("occluded", 0)))
        ET.SubElement(obj, "difficult").text = str(int(det.get("difficult", 0)))
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(round(x * width))
        ET.SubElement(box, "ymin").text = str(round(y * height))
        ET.SubElement(box, "xmax").text = str(round((x + w) * width))
        ET.SubElement(box, "ymax").text = str(round((y + h) * height))
        n_objects += 1

    ET.indent(root, space="    ")
    dest = os.path.join(ANNOTATIONS, name.replace(".png", ".xml"))
    ET.ElementTree(root).write(dest, encoding="utf-8", xml_declaration=False)
    return n_objects


def write_annotations(samples):
    os.makedirs(ANNOTATIONS, exist_ok=True)
    total = sum(to_voc(sample) for sample in samples)
    print(f"Wrote {len(samples)} Pascal VOC files to {ANNOTATIONS}/  ({total:,} boxes)")
    return total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = download_samples()
    assert len(samples) == EXPECTED_IMAGES, f"mirror returned {len(samples)} samples"

    download_images(samples)
    n_images = len([f for f in os.listdir(IMAGES) if f.endswith(".png")])
    assert n_images == EXPECTED_IMAGES, f"expected {EXPECTED_IMAGES} images, found {n_images}"

    total = write_annotations(samples)
    assert total == EXPECTED_OBJECTS, f"expected {EXPECTED_OBJECTS} boxes, wrote {total}"

    size_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(OUT_DIR)
        for f in files
    ) / 1e6
    print(f"\nReady: {n_images:,} images + {n_images:,} annotations in {OUT_DIR}/ ({size_mb:.0f} MB)")
    print("Now run: python prepare_dataset.py   (VOC -> YOLO, then train/val/test split)")


if __name__ == "__main__":
    main()
