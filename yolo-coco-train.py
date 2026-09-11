import os

ROOT = os.path.dirname(__file__)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".yolo_config"))

from ultralytics import YOLO

if __name__ == "__main__":
    # Baseline sanity check: run the migrated SpikeYOLO model first and verify KITTI mAP is non-zero.
    model = YOLO("ultralytics/cfg/models/26/yolo26-snn-closed.yaml")
    model.train(
        data="yolo-cocoval-data.yaml",
        workers=2,
        epochs=30,
        batch=16,
        project=os.path.join(ROOT, "runs", "Coco"),
        name="26-snn-closed",
        device=0,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.002,
        weight_decay=0.025,
        warmup_bias_lr=0.0,
        close_mosaic=2,
    )

