from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".reviewer_config"))

from ultralytics import YOLOE
from ultralytics.nn.modules.head import omnispike_SNNSegment22_2DirectTextRouteSER


MODEL_YAML = Path(__file__).resolve().parent / "ultralytics/cfg/models/26/omnispike-text-route-bira-reg16.yaml"


def main() -> None:
    wrapper = YOLOE(MODEL_YAML, task="segment", verbose=False)
    core = wrapper.model
    head = core.model[-1]

    assert isinstance(head, omnispike_SNNSegment22_2DirectTextRouteSER)
    assert head.reg_max == 16
    assert head.end2end is True

    core.train()
    features = core.predict(
        torch.randn(1, 3, 64, 64),
        tpe=torch.randn(1, 5, 512),
    )
    assert set(features) == {"one2many", "one2one"}
    assert "proto" in features["one2many"]
    assert "proto" in features["one2one"]

    print("Model construction and forward verification passed.")
    print(f"Head: {type(head).__name__}")
    print(f"Boxes: {tuple(features['one2many']['boxes'].shape)}")
    print(f"Proto: {tuple(features['one2many']['proto'].shape)}")


if __name__ == "__main__":
    main()

