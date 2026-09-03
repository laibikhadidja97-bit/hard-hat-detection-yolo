"""
Gradio demo: drop in a site photo, get a hard-hat compliance verdict.

    python app.py                       # http://127.0.0.1:7860
    python app.py --weights path/to/best.pt --share

The interface takes an upload or a webcam frame, runs the trained detector, and
answers the operational question directly — how many heads are protected, how
many are not — instead of only drawing boxes.
"""
import argparse
import glob
import os

import gradio as gr
from ultralytics import YOLO

DEFAULT_WEIGHTS = "results/best.pt"


def find_weights(path):
    if os.path.exists(path):
        return path
    candidates = sorted(glob.glob("runs/detect/*/weights/best.pt"), key=os.path.getmtime)
    if not candidates:
        raise SystemExit(
            f"No weights at {path} and nothing in runs/detect/*/weights/best.pt — "
            "train first with: python train.py"
        )
    return candidates[-1]


def build(model):
    def detect(image, conf, iou):
        if image is None:
            return None, "Upload a photo or take a webcam shot to start."
        result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
        counts = {}
        for cls in result.boxes.cls.tolist():
            name = model.names[int(cls)]
            counts[name] = counts.get(name, 0) + 1
        helmets, heads = counts.get("helmet", 0), counts.get("head", 0)
        people = helmets + heads

        if people == 0:
            verdict = "### No heads detected\nNothing to assess in this frame."
        elif heads == 0:
            verdict = (f"### ✅ Compliant\n**{helmets}** of **{people}** heads are wearing a "
                       f"hard hat.")
        else:
            verdict = (f"### ⚠️ {heads} violation{'s' if heads > 1 else ''}\n"
                       f"**{heads}** of **{people}** heads are unprotected "
                       f"({helmets} wearing a hard hat).")
        rate = f"\n\nCompliance rate: **{helmets / people:.0%}**" if people else ""
        return result.plot(line_width=2)[:, :, ::-1], verdict + rate

    with gr.Blocks(title="Hard Hat Detection") as demo:
        gr.Markdown(
            "# Hard Hat Detection\n"
            "YOLOv8 fine-tuned on 5,000 annotated construction-site images. "
            "Green boxes are protected heads, red boxes are safety violations."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Site photo", sources=["upload", "clipboard", "webcam"],
                                 type="numpy", height=380)
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                iou = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label="NMS IoU threshold")
                run = gr.Button("Detect", variant="primary")
            with gr.Column():
                output = gr.Image(label="Detections", height=380)
                verdict = gr.Markdown()
        examples = sorted(glob.glob("results/examples/*.png"))[:6]
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[image], label="Test-set examples")
        run.click(detect, [image, conf, iou], [output, verdict])
        image.change(detect, [image, conf, iou], [output, verdict])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--share", action="store_true", help="expose a temporary public link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    weights = find_weights(args.weights)
    print(f"Loading {weights}")
    build(YOLO(weights)).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
