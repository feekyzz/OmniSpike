from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "omnispike-yolo-config"))

from ultralytics import YOLOE  # noqa: E402


WEIGHTS = ROOT / "weights" / "omnispike-text-route-ser-best.pt"
DEFAULT_DATA = ROOT / "yolo-coco128-rectseg-smoke-data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniSpike segmentation validation.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline-smoke", action="store_true", help="Use deterministic random text features and skip MobileCLIP download.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(tempfile.gettempdir()) / "omnispike-reviewer" / "val",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. Pass --data with the reviewer's local dataset YAML."
        )

    model = YOLOE(args.weights, task="segment", verbose=False)
    if args.offline_smoke:
        def offline_get_text_pe(names, batch=80, cache_clip_model=False, without_reprta=False):
            embedding_dim = int(getattr(model.model.model[-1], "embed", 512))
            return torch.randn(
                1,
                len(names),
                embedding_dim,
                device=next(model.model.parameters()).device,
                generator=torch.Generator().manual_seed(0),
            )

        model.model.get_text_pe = offline_get_text_pe
    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=False,
        save_json=False,
        project=str(args.project),
        name="val",
        exist_ok=True,
        verbose=False,
    )
    results = getattr(metrics, "results_dict", {})
    mode = "offline-random-text" if args.offline_smoke else "mobileclip-text"
    print(f"val_ok mode={mode} head={model.model.model[-1].__class__.__name__} metrics={results}")


if __name__ == "__main__":
    main()
