from __future__ import annotations

import argparse
import os
import tempfile
from copy import copy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "omnispike-yolo-config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainer, YOLOESegValidator  # noqa: E402


WEIGHTS = ROOT / "weights" / "omnispike-text-route-ser-best.pt"
DEFAULT_DATA = ROOT / "yolo-coco128-rectseg-smoke-data.yaml"


class OfflineSmokeValidator(YOLOESegValidator):
    """Validation adapter that avoids downloading the optional MobileCLIP text encoder."""

    def __call__(self, trainer=None, model=None, refer_data=None, load_vp=False):
        if trainer is None and isinstance(model, (str, Path)):
            from ultralytics.nn.tasks import load_checkpoint
            from ultralytics.utils.torch_utils import select_device

            device = select_device(self.args.device, verbose=False)
            model, _ = load_checkpoint(model, device=device)
        target = trainer.ema.ema if trainer is not None else model
        had_instance_method = target is not None and "get_text_pe" in target.__dict__
        original_method = target.__dict__.get("get_text_pe") if had_instance_method else None
        if target is not None:
            def offline_get_text_pe(names, batch=80, cache_clip_model=False, without_reprta=False):
                embedding_dim = int(getattr(target.model[-1], "embed", 512))
                generator = torch.Generator(device="cpu").manual_seed(0)
                return torch.randn(1, len(names), embedding_dim, generator=generator).to(
                    next(target.parameters()).device
                )
            target.get_text_pe = offline_get_text_pe
        try:
            return super().__call__(trainer=trainer, model=model, refer_data=refer_data, load_vp=load_vp)
        finally:
            if target is not None:
                if had_instance_method:
                    target.get_text_pe = original_method
                else:
                    del target.get_text_pe


class OfflineSmokeTrainer(YOLOESegTrainer):
    """Trainer variant used only for a no-download pipeline smoke test."""

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return OfflineSmokeValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal OmniSpike segmentation training smoke test.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline-smoke", action="store_true", help="Use random text features and skip MobileCLIP download.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(tempfile.gettempdir()) / "omnispike-reviewer" / "train",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. Pass --data with the reviewer's local dataset YAML."
        )

    model = YOLOE(args.weights, task="segment", verbose=False)
    trainer_cls = OfflineSmokeTrainer if args.offline_smoke else YOLOESegTrainer
    model.train(
        data=str(args.data),
        trainer=trainer_cls,
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer="AdamW",
        lr0=1e-4,
        warmup_bias_lr=0.0,
        amp=False,
        cache=False,
        val=False,
        plots=False,
        save=True,
        project=str(args.project),
        name="train",
        exist_ok=True,
    )
    mode = "offline-random-text" if args.offline_smoke else "mobileclip-text"
    print(f"train_ok mode={mode} head={model.model.model[-1].__class__.__name__} output={args.project / 'train'}")


if __name__ == "__main__":
    main()
