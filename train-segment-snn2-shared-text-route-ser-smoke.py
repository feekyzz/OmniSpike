from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainerFromScratch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="共享文本路由SNN分割Head的两轮小数据训练测试。")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", choices=("n", "s"), default="s")
    args = parser.parse_args()

    model_yaml = ROOT / f"ultralytics/cfg/models/26/yoloe-26{args.scale}-snn2-seg22-2-shared-text-route-ser.yaml"
    data_yaml = ROOT / "yolo-coco128-rectseg-smoke-data.yaml"
    data = {
        "train": {"yolo_data": [str(data_yaml)]},
        "val": {"yolo_data": [str(data_yaml)]},
    }
    model = YOLOE(model_yaml, task="segment", verbose=True)
    model.train(
        data=data,
        trainer=YOLOESegTrainerFromScratch,
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=args.device,
        optimizer="AdamW",
        lr0=1e-3,
        warmup_bias_lr=0.0,
        pretrained=False,
        amp=False,
        cache=False,
        val=False,
        plots=False,
        save=True,
        project=str(ROOT / "runs/smoke-segment-shared-text-route-ser"),
        name=f"snn2-seg22-2-shared-text-route-ser-direct-base-{args.scale}-2epoch",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
