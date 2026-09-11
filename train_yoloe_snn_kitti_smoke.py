from pathlib import Path

import torch

from ultralytics import YOLOE


def main():
    repo_root = Path(__file__).resolve().parent
    data_yaml = repo_root / "yolo-KITTI-smoke-data.yaml"
    model_yaml = repo_root / "ultralytics" / "cfg" / "models" / "26" / "yoloe-26-snn.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    if not model_yaml.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")

    # 中文说明：
    # 这个脚本只用于快速验证“YOLOE-SNN 训练链能不能跑通”，
    # 所以这里故意使用一个很小的 KITTI 子集，并把 epoch 固定为 3。
    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = "yoloe-26-snn-kitti-smoke"

    model = YOLOE(str(model_yaml))
    results = model.train(
        data=str(data_yaml),
        epochs=3,
        imgsz=224,
        batch=2,
        workers=0,
        device=device,
        optimizer="AdamW",
        lr0=1e-3,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        close_mosaic=0,
        amp=False,
        cache=False,
        pretrained=False,
        val=True,
        verbose=True,
        deterministic=True,
        project=str(repo_root / "runs" / "smoke"),
        name=run_name,
        exist_ok=True,
    )

    # The final directory is useful when checking logs/weights after a quick smoke run.
    save_dir = getattr(results, "save_dir", None)
    print(f"Smoke training finished. save_dir={save_dir}")


if __name__ == "__main__":
    main()
