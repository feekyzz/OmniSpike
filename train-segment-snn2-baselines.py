from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainerFromScratch  # noqa: E402


MODEL_YAMLS = {
    "20": ROOT / "ultralytics/cfg/models/26/yoloe-26-snn2-seg20.yaml",
    "21": ROOT / "ultralytics/cfg/models/26/yoloe-26-snn2-seg21.yaml",
    "22": ROOT / "ultralytics/cfg/models/26/yoloe-26-snn2-seg22.yaml",
}


def prepare_smoke_dataset(max_images: int = 4) -> Path:
    """Convert a few local COCO128 boxes to rectangular polygons for pipeline-only smoke training."""
    src_root = ROOT / "datasets/coco128"
    dst_root = ROOT / "datasets/coco128-rectseg-smoke"
    src_images = src_root / "images/train2017"
    src_labels = src_root / "labels/train2017"
    dst_images = dst_root / "images/train2017"
    dst_labels = dst_root / "labels/train2017"
    yaml_path = ROOT / "yolo-coco128-rectseg-smoke-data.yaml"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    images = [p for p in sorted(src_images.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise FileNotFoundError(f"No smoke-test images found under {src_images}")

    for image_path in images[:max_images]:
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.copy2(image_path, dst_images / image_path.name)
        polygons = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                polygons.append(raw.strip())
                continue
            cls, x, y, w, h = parts
            x, y, w, h = map(float, (x, y, w, h))
            x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
            x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
            polygons.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
        (dst_labels / label_path.name).write_text("\n".join(polygons) + ("\n" if polygons else ""), encoding="utf-8")

    names = (ROOT / "yolo-coco128-data.yaml").read_text(encoding="utf-8").split("# Classes", 1)[1]
    yaml_path.write_text(
        "# 仅用于验证训练链：bbox 被转换为矩形 polygon，不用于正式分割精度实验。\n"
        "path: datasets/coco128-rectseg-smoke\n"
        "train: images/train2017\n"
        "val: images/train2017\n\n"
        "# Classes"
        + names,
        encoding="utf-8",
    )
    return yaml_path


def train_variant(variant: str, data_yaml: Path, epochs: int, imgsz: int, batch: int, device: str):
    model = YOLOE(MODEL_YAMLS[variant], task="segment", verbose=True)
    data = {
        "train": {"yolo_data": [str(data_yaml)]},
        "val": {"yolo_data": [str(data_yaml)]},
    }
    return model.train(
        data=data,
        trainer=YOLOESegTrainerFromScratch,
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=device,
        optimizer="AdamW",
        lr0=1e-3,
        warmup_bias_lr=0.0,
        pretrained=False,
        amp=False,
        cache=False,
        val=False,
        plots=False,
        save=False,
        project=str(ROOT / "runs/smoke-segment-baselines"),
        name=f"snn2-seg{variant}-2epoch",
        exist_ok=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-train YOLOE-SNN2 segmentation 2.0/2.1/2.2 baselines.")
    parser.add_argument("--variant", choices=("20", "21", "22", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    smoke_yaml = prepare_smoke_dataset(args.max_images)
    variants = MODEL_YAMLS if args.variant == "all" else (args.variant,)
    for current_variant in variants:
        train_variant(current_variant, smoke_yaml, args.epochs, args.imgsz, args.batch, args.device)
