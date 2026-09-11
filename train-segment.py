import os
import shutil
from pathlib import Path

ROOT = os.path.dirname(__file__)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".yolo_config"))

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainerFromScratch


def prepare_coco128_rectseg() -> str:
    """Create a small pseudo segmentation dataset from coco128 boxes.

    This is only for checking that the SNN open-vocabulary segmentation training
    pipeline runs. It converts each bbox into a rectangular polygon and writes
    the result to a separate dataset folder, leaving the original coco128 labels
    untouched. Replace it with a real segmentation dataset for actual results.
    """

    src_root = Path(ROOT) / "datasets" / "coco128"
    dst_root = Path(ROOT) / "datasets" / "coco128-rectseg"
    src_img_dir = src_root / "images" / "train2017"
    src_label_dir = src_root / "labels" / "train2017"
    dst_img_dir = dst_root / "images" / "train2017"
    dst_label_dir = dst_root / "labels" / "train2017"
    yaml_path = Path(ROOT) / "yolo-coco128-rectseg-data.yaml"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_img_dir.glob("*.*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            dst_img = dst_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

    for label_path in src_label_dir.glob("*.txt"):
        out_lines = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                # Keep existing segmentation labels if the source already has them.
                out_lines.append(raw.strip())
                continue
            cls, x, y, w, h = parts
            x, y, w, h = map(float, (x, y, w, h))
            x1 = max(0.0, x - w / 2)
            y1 = max(0.0, y - h / 2)
            x2 = min(1.0, x + w / 2)
            y2 = min(1.0, y + h / 2)
            out_lines.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
        (dst_label_dir / label_path.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    src_yaml = (Path(ROOT) / "yolo-coco128-data.yaml").read_text(encoding="utf-8")
    src_yaml = src_yaml.replace("path: 'datasets/coco128'", "path: 'datasets/coco128-rectseg'")
    src_yaml = src_yaml.replace("path: datasets/coco128", "path: datasets/coco128-rectseg")
    yaml_path.write_text(
        "# Auto-generated pseudo segmentation yaml. Rectangular masks are converted from bbox labels.\n" + src_yaml,
        encoding="utf-8",
    )
    return str(yaml_path)


if __name__ == "__main__":
    # Keep the same multi-modal data wrapper style as yolo-KITTI-train.py.
    # For real experiments, replace SEG_DATA_YAML with your true segmentation
    # dataset yaml instead of using the auto-generated rectangular pseudo masks.
    SEG_DATA_YAML = os.environ.get("SEG_DATA_YAML") or prepare_coco128_rectseg()
    data = dict(
        train=dict(yolo_data=[SEG_DATA_YAML]),
        val=dict(yolo_data=[SEG_DATA_YAML]),
    )

    model = YOLOE("ultralytics/cfg/models/26/yoloe-26-snn2-seg-text-ser.yaml")
    model.train(
        data=data,
        workers=0,
        epochs=30,
        batch=4,
        project=os.path.join(ROOT, "runs", "SEGMENT"),
        name="snn-seg-text-ser-1",
        device=0,
        trainer=YOLOESegTrainerFromScratch,
        pretrained=False,
    )
