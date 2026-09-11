import os

ROOT = os.path.dirname(__file__)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".yolo_config"))

from ultralytics import YOLOE


def main():
    model = YOLOE(r"runs/Coco-yoloe/yoloe-26-snn3/weights/last.pt")
    # model.model.text_model = "mobileclip:blt"
    model.val(
        data="yolo-cocoval-data.yaml",
        batch=1,
        split="val",
        # rect=False,
        # max_det=1000,
        save_json=True,
        device=0,
        name="snn-val",
        workers=0,
    )


if __name__ == "__main__":
    main()
