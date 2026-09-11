from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "omnispike-yolo-config"))

from ultralytics import YOLOE  # noqa: E402


WEIGHTS = ROOT / "weights" / "omnispike-text-route-ser-best.pt"


def _classes(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--classes must contain at least one class name")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniSpike text-prompt segmentation prediction.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--source", type=Path, default=None, help="Image/video/directory; omitted uses a synthetic image.")
    parser.add_argument("--classes", default="object", help="Comma-separated text prompts, e.g. person,car.")
    parser.add_argument("--offline-smoke", action="store_true", help="Use deterministic random text features and skip MobileCLIP download.")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", action="store_true", help="Save predictions to --project instead of keeping the test in memory.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(tempfile.gettempdir()) / "omnispike-reviewer" / "predict",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.source is not None and not args.source.exists():
        raise FileNotFoundError(f"Prediction source not found: {args.source}")

    classes = _classes(args.classes)
    model = YOLOE(args.weights, task="segment", verbose=False)
    if args.offline_smoke:
        embedding_dim = int(getattr(model.model.model[-1], "embed", 512))
        embeddings = torch.randn(1, len(classes), embedding_dim, device="cpu", generator=torch.Generator().manual_seed(0))
        model.model.set_classes(classes, embeddings)
        model.model.names = classes
    else:
        model.set_classes(classes)
    source = args.source if args.source is not None else np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        project=str(args.project),
        name="predict",
        exist_ok=True,
        verbose=False,
    )
    first = results[0] if results else None
    boxes = int(len(first.boxes)) if first is not None and first.boxes is not None else 0
    masks = int(len(first.masks)) if first is not None and first.masks is not None else 0
    mode = "offline-random-text" if args.offline_smoke else "mobileclip-text"
    print(f"predict_ok mode={mode} results={len(results)} boxes={boxes} masks={masks} head={model.model.model[-1].__class__.__name__}")


if __name__ == "__main__":
    main()
