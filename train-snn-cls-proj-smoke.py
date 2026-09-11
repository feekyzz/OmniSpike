"""Train the spike-driven cls_proj detection and segmentation heads on a four-image smoke dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch  # noqa: E402
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainerFromScratch  # noqa: E402


DATA_YAML = ROOT / "yolo-coco128-rectseg-smoke-data.yaml"
RUNS_DIR = ROOT / "runs" / "smoke-snn-cls-proj"
MODEL_STEMS = {
    "detect": "yoloe-26{scale}-snn2-1.yaml",
    "segment": "yoloe-26{scale}-snn2-seg22-1.yaml",
    "segment2": "yoloe-26{scale}-snn2-seg22-2.yaml",
    "segment_ser": "yoloe-26{scale}-snn2-seg22-2-res-text-ser.yaml",
}


def train(task: str, epochs: int, scale: str, weights: str | None, suffix: str) -> None:
    """Train one smoke model and retain last.pt/best.pt for operation profiling."""
    runtime_task = "detect" if task == "detect" else "segment"
    trainer = YOLOETrainerFromScratch if runtime_task == "detect" else YOLOESegTrainerFromScratch
    model_yaml = ROOT / "ultralytics/cfg/models/26" / MODEL_STEMS[task].format(scale=scale)
    model = YOLOE(model_yaml, task=runtime_task, verbose=True)
    if weights:
        model.load(weights)
    data = {
        "train": {"yolo_data": [str(DATA_YAML)]},
        "val": {"yolo_data": [str(DATA_YAML)]},
    }
    model.train(
        data=data,
        trainer=trainer,
        task=runtime_task,
        epochs=epochs,
        imgsz=64,
        batch=2,
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        optimizer="AdamW",
        lr0=1e-3,
        warmup_bias_lr=0.0,
        pretrained=False,
        amp=False,
        cache=False,
        val=False,
        plots=False,
        save=True,
        save_period=-1,
        mosaic=0.0,
        project=str(RUNS_DIR),
        name=f"{task}-{suffix}",
        exist_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detect", "segment", "segment2", "segment_ser", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--scale", choices=("n", "s", "m", "l"), default="n")
    parser.add_argument("--detect-weights")
    parser.add_argument("--segment-weights")
    parser.add_argument("--segment2-weights")
    parser.add_argument("--segment-ser-weights")
    parser.add_argument("--suffix", default="2epoch")
    args = parser.parse_args()
    tasks = ("detect", "segment") if args.task == "all" else (args.task,)
    initial_weights = {
        "detect": args.detect_weights,
        "segment": args.segment_weights,
        "segment2": args.segment2_weights,
        "segment_ser": args.segment_ser_weights,
    }
    for current_task in tasks:
        train(current_task, args.epochs, args.scale, initial_weights[current_task], args.suffix)
