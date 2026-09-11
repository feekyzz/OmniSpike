from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.nn.modules.head import YOLOESNNSegment22_2, omnispike_SNNSegment22_2DirectTextRouteSER  # noqa: E402
from ultralytics.nn.modules.yolo_spikformer import MS_StandardConv, SpikeConv, SpikeConvWithoutBN  # noqa: E402


def _is_multispike4(tensor: torch.Tensor) -> bool:
    return bool(
        tensor.numel()
        and tensor.min().item() >= 0.0
        and tensor.max().item() <= 4.0
        and torch.allclose(tensor, tensor.round(), atol=0.0, rtol=0.0)
    )


def audit_direct_text_route_ser_spikes(model_source: str | Path) -> dict[str, object]:
    """审计直接替换结构、三分支共享关系，以及所有SNN卷积前的MultiSpike4发放。"""
    wrapper = YOLOE(model_source, task="segment", verbose=False)
    core = wrapper.model
    head = core.model[-1]
    assert isinstance(head, omnispike_SNNSegment22_2DirectTextRouteSER)
    assert head.__class__.__bases__ == (YOLOESNNSegment22_2,)
    assert not hasattr(head.direct_text_ser, "p4_scale_logit")
    assert not hasattr(head.direct_text_ser, "p5_scale_logit")

    direct: list[torch.Tensor] = []
    branch_inputs: dict[str, dict[int, torch.Tensor]] = {"box": {}, "cls": {}, "mask": {}}
    handles = [head.direct_text_ser.register_forward_hook(lambda _module, _inputs, outputs: direct.extend(outputs))]

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
        # MS_ConvBlock内部由SepConv/RepConv直接持有Conv2d，需要逐个审计真实卷积输入。
        elif isinstance(module, torch.nn.Conv2d) and ("p4_refine" in name or "p5_refine" in name):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs, name=name: conv_inputs.setdefault(name, inputs[0].detach().cpu())
                )
            )

    core.train()
    with torch.no_grad():
        outputs = core.predict(torch.randn(1, 3, 64, 64), tpe=torch.randn(1, 5, 512))
    for handle in handles:
        handle.remove()

    assert set(outputs) == {"one2many", "one2one"}
    assert len(direct) == 3
    for level in range(3):
        for branch in ("box", "cls", "mask"):
            assert branch_inputs[branch][level].data_ptr() == direct[level].data_ptr()
    invalid = {
        name: (value.min().item(), value.max().item())
        for name, value in conv_inputs.items()
        if not _is_multispike4(value)
    }
    assert not invalid, f"发现未经过MultiSpike4发放的SNN卷积输入: {invalid}"
    def summarize_firing(name_parts: tuple[str, ...]) -> dict[str, float | int]:
        rates = [float((value != 0).float().mean()) for name, value in conv_inputs.items() if any(p in name for p in name_parts)]
        return {
            "count": len(rates),
            "mean": sum(rates) / len(rates) if rates else 0.0,
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
        }

    return {
        "snn_conv_count": len(conv_inputs),
        "spike_values": sorted({float(v) for x in conv_inputs.values() for v in torch.unique(x).tolist()}),
        "direct_p4_is_integer": _is_multispike4(direct[1].detach().cpu()),
        "direct_p5_is_integer": _is_multispike4(direct[2].detach().cpu()),
        "branches_share_same_tensor": True,
        "external_residual_or_scale": False,
        "firing_rate": {
            "text_ser": summarize_firing(("p4_text_router", "p5_text_router")),
            "post_attention_refine": summarize_firing(("p4_refine", "p5_refine")),
            "head_branches": summarize_firing((".cv2.", ".cv3.", ".cv5.")),
        },
    }


def test_direct_text_route_ser_head_and_spike_firing() -> None:
    yaml_path = ROOT / "ultralytics/cfg/models/26/yoloe-26n-snn2-seg22-2-direct-text-route-ser.yaml"
    audit_direct_text_route_ser_spikes(yaml_path)
