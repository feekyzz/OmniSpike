from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.nn.modules.head import (  # noqa: E402
    YOLOESNNSegment22_2,
    YOLOESNNSegment22_2SharedTextRouteSER,
)
from ultralytics.nn.modules.yolo_spikformer import (  # noqa: E402
    MS_StandardConv,
    SpikeConv,
    SpikeConvWithoutBN,
)


def _is_multispike4(tensor: torch.Tensor) -> bool:
    return bool(
        tensor.numel()
        and tensor.min().item() >= 0.0
        and tensor.max().item() <= 4.0
        and torch.allclose(tensor, tensor.round(), atol=0.0, rtol=0.0)
    )


def audit_shared_text_route_ser_spikes(model_source: str | Path) -> dict[str, object]:
    """审计共享P4/P5进入三个分支，且所有SNN卷积在卷积前完成MultiSpike4发放。"""
    wrapper = YOLOE(model_source, task="segment", verbose=False)
    core = wrapper.model
    head = core.model[-1]
    assert isinstance(head, YOLOESNNSegment22_2SharedTextRouteSER)
    assert head.__class__.__bases__ == (YOLOESNNSegment22_2,)

    shared: list[torch.Tensor] = []
    branch_inputs: dict[str, dict[int, torch.Tensor]] = {"box": {}, "cls": {}, "mask": {}}
    handles = []

    def capture_shared(_module, _inputs, outputs):
        shared[:] = outputs

    handles.append(head.shared_enhancer.register_forward_hook(capture_shared))

    def capture_branch(branch: str, level: int):
        def hook(_module, inputs):
            branch_inputs[branch].setdefault(level, inputs[0])

        return hook

    for level in range(3):
        handles.append(head.cv2[level][0].register_forward_pre_hook(capture_branch("box", level)))
        handles.append(head.cv3[level][0].register_forward_pre_hook(capture_branch("cls", level)))
        handles.append(head.cv5[level].register_forward_pre_hook(capture_branch("mask", level)))

    conv_inputs: dict[str, torch.Tensor] = {}
    for name, module in core.named_modules():
        if isinstance(module, (MS_StandardConv, SpikeConv, SpikeConvWithoutBN)):
            handles.append(
                module.conv.register_forward_pre_hook(
                    lambda _module, inputs, name=name: conv_inputs.setdefault(name, inputs[0].detach().cpu())
                )
            )

    core.train()
    image = torch.randn(1, 3, 64, 64)
    text = torch.randn(1, 5, 512)
    with torch.no_grad():
        outputs = core.predict(image, tpe=text)

    for handle in handles:
        handle.remove()
    assert set(outputs) == {"one2many", "one2one"}
    assert len(shared) == 3
    for level in range(3):
        for branch in ("box", "cls", "mask"):
            # 三条Head路径必须接收同一个共享Tensor对象，而不是各自重新计算注意力。
            assert branch_inputs[branch][level].data_ptr() == shared[level].data_ptr()
    assert conv_inputs
    invalid = {
        name: (value.min().item(), value.max().item())
        for name, value in conv_inputs.items()
        if not _is_multispike4(value)
    }
    assert not invalid, f"发现未经过MultiSpike4发放的SNN卷积输入: {invalid}"
    all_unique = sorted({float(item) for value in conv_inputs.values() for item in torch.unique(value).tolist()})
    return {
        "snn_conv_count": len(conv_inputs),
        "spike_values": all_unique,
        "shared_p4_is_integer": _is_multispike4(shared[1].detach().cpu()),
        "shared_p5_is_integer": _is_multispike4(shared[2].detach().cpu()),
        "branches_share_same_tensor": True,
    }


def test_shared_text_route_ser_head_and_spike_firing() -> None:
    yaml_path = ROOT / "ultralytics/cfg/models/26/yoloe-26n-snn2-seg22-2-shared-text-route-ser.yaml"
    audit_shared_text_route_ser_spikes(yaml_path)
