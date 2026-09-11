import os

ROOT = os.path.dirname(__file__)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".yolo_config"))

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch


def main():
    data = dict(
        train=dict(yolo_data=["yolo-cocoval-data.yaml"]),
        val=dict(yolo_data=["yolo-cocoval-data.yaml"]),
    )

    model = YOLOE("runs/Coco-yoloe/yoloe-26-snn-earlyann/weights/best.pt")

    model.train(
        data=data,
        name="yoloe-26-snn-earlyann-freeze-backbone-test",
        batch=16,
        epochs=30,
        close_mosaic=2,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_bias_lr=0.0,
        weight_decay=0.025,
        momentum=0.9,
        workers=2,
        trainer=YOLOETrainerFromScratch,
        project=os.path.join(ROOT, "runs", "Coco-yoloe"),
        device=0,
        freeze=10,
    )


if __name__ == "__main__":
    main()