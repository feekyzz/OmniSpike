import argparse
import os

ROOT = os.path.dirname(__file__)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".yolo_config"))

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="yolo-coco3-data.yaml")
    parser.add_argument("--model", default="ultralytics/cfg/models/26/yoloe-26-snn2.yaml")
    parser.add_argument("--name", default="yoloe-26-snn2-coco3")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data = dict(
        train=dict(yolo_data=[args.data]),
        val=dict(yolo_data=[args.data]),
    )

    model = YOLOE(args.model)
    model.train(
        data=data,
        name=args.name,
        batch=16,
        epochs=args.epochs,
        close_mosaic=2,
        optimizer="AdamW",
        lr0=2e-3,
        warmup_bias_lr=0.0,
        weight_decay=0.025,
        momentum=0.9,
        workers=2,
        trainer=YOLOETrainerFromScratch,
        project=os.path.join(ROOT, "runs", "Coco-yoloe-subset"),
        device=0,
        pretrained=False,
    )


if __name__ == "__main__":
    main()
