# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import hashlib
from copy import copy, deepcopy

import torch

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import LOGGER, RANK

from .train import YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from .val import YOLOESegValidator


def text_adapter_fingerprint(head) -> str:
    """Hash the learned text adapter that turns raw text features into model-specific prompt embeddings."""
    if not hasattr(head, "reprta"):
        raise AttributeError(f"{type(head).__name__} does not expose the reprta text adapter.")
    digest = hashlib.sha256()
    for name, tensor in sorted(head.reprta.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def verify_coco_transfer_equivalence(head, text_features: torch.Tensor, atol: float = 5e-4) -> float:
    """Verify that every fixed COCO classifier reproduces its pretrained contrastive head before training."""
    branch_pairs = [(head.cv4, head.transfer_cls)]
    if getattr(head, "end2end", False):
        branch_pairs.append((head.one2one_cv4, head.one2one_transfer_cls))

    max_error = 0.0
    for contrastive_heads, transfer_heads in branch_pairs:
        for contrastive_head, transfer_head in zip(contrastive_heads, transfer_heads):
            channels = transfer_head.in_channels
            sample = torch.linspace(
                -1.0,
                1.0,
                steps=channels * 4,
                device=transfer_head.weight.device,
                dtype=transfer_head.weight.dtype,
            ).reshape(1, channels, 2, 2)
            sample = torch.nn.functional.normalize(sample, dim=1, eps=1e-6)
            expected = contrastive_head(sample, text_features.to(sample.device, sample.dtype))
            actual = transfer_head(sample)
            max_error = max(max_error, float((expected - actual).abs().max()))

    if max_error > atol:
        raise RuntimeError(
            f"COCO transfer classifier does not reproduce the pretrained contrastive logits: "
            f"max_abs_error={max_error:.6g}, atol={atol:.6g}."
        )
    return max_error


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """Trainer class for YOLOE segmentation models.

    This class combines YOLOETrainer and SegmentationTrainer to provide training functionality specifically for YOLOE
    segmentation models, enabling both object detection and instance segmentation capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with training parameters.
        overrides (dict): Dictionary with parameter overrides.
        _callbacks (list): List of callback functions for training events.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOESegModel initialized with specified config and weights.

        Args:
            cfg (dict | str, optional): Model configuration dictionary or YAML file path.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (YOLOESegModel): Initialized YOLOE segmentation model.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Create and return a validator for YOLOE segmentation model evaluation.

        Returns:
            (YOLOESegValidator): Validator for YOLOE segmentation models.
        """
        # 当前 v8SegmentationLoss 固定返回 box/seg/cls/dfl/sem 五项；即使没有语义分割分支，sem 也是零值占位。
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return YOLOESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class YOLOEPESegTrainer(SegmentationTrainer):
    """Fine-tune YOLOESeg model in linear probing way.

    This trainer specializes in fine-tuning YOLOESeg models using a linear probing approach, which involves freezing
    most of the model and only training specific layers for efficient adaptation to new tasks.

    Attributes:
        data (dict): Dataset configuration containing channels, class names, and number of classes.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOESegModel initialized with specified config and weights for linear probing.

        Args:
            cfg (dict | str, optional): Model configuration dictionary or YAML file path.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (YOLOESegModel): Initialized YOLOE segmentation model configured for linear probing.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # NOTE: `get_text_pe` related to text model and YOLOEDetect.reprta,
        # it'd get correct results as long as loading proper pretrained weights.
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        model.train()

        return model


class YOLOECOCOTransferSegTrainer(SegmentationTrainer):
    """Closed-set COCO downstream transfer trainer adapted to the spike-driven YOLOE classification projection."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load pretrained weights, fixed COCO prompts, and create the transfer-only classification readout."""
        model = YOLOESegModel(
            # 权重已经保存了完整模型结构，直接使用结构字典，避免 yaml_file 中的服务器绝对路径导致权重无法跨机器训练。
            cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        assert weights is not None, "COCO downstream transfer requires pretrained weights."

        pe_path = getattr(self.args, "train_pe_path", None)
        assert pe_path, "COCO downstream transfer requires train_pe_path."
        pe_state = torch.load(pe_path, map_location="cpu")
        names = list(self.data["names"].values())
        if pe_state["names"] != names:
            raise ValueError("The cached prompt names/order do not match the COCO dataset names.")

        # 首次迁移必须先加载开放词汇预训练权重，再根据已经训练好的cv4构造等价固定分类器。
        # 旧顺序在随机初始化模型上提前构造transfer_cls，原始checkpoint又不含这些新模块，
        # 导致Linear Probing第一轮分类指标接近零。
        source_model = weights["model"] if isinstance(weights, dict) else weights
        source_head = source_model.model[-1]
        source_has_transfer = hasattr(source_head, "transfer_cls")
        model.load(weights)

        model.eval()
        head = model.model[-1]
        cached_fingerprint = pe_state.get("text_adapter_fingerprint")
        loaded_fingerprint = text_adapter_fingerprint(head)
        if cached_fingerprint != loaded_fingerprint:
            raise ValueError(
                "The COCO prompt cache was not generated by this checkpoint's text adapter. "
                "Delete the stale cache or omit --pe to use the model-specific default cache."
            )
        model.set_classes(pe_state["names"], pe_state["pe"])
        if not hasattr(head, "prepare_coco_transfer"):
            raise TypeError(f"{type(head).__name__} does not implement prepare_coco_transfer().")
        head.prepare_coco_transfer(model.pe)
        equivalence_error = verify_coco_transfer_equivalence(head, model.pe)
        LOGGER.info(
            f"COCO transfer initialization verified against pretrained contrastive logits "
            f"(max_abs_error={equivalence_error:.3g})."
        )
        # 若输入本身是迁移学习checkpoint，第二次加载可恢复其中已有的transfer_cls；
        # 原始开放词汇checkpoint没有这些模块，因此无需二次加载。
        if source_has_transfer:
            model.load(weights)
        model.train()
        return model

    def _setup_train(self):
        """Run the normal setup, then fail fast if linear probing exposes any non-classifier parameter."""
        super()._setup_train()
        mode = getattr(self.args, "transfer_mode", None)
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        LOGGER.info(
            f"COCO transfer mode={mode}: {len(trainable)} trainable tensors, {parameter_count:,} trainable parameters."
        )
        if mode == "linear":
            allowed = (".transfer_cls.", ".one2one_transfer_cls.")
            unexpected = [name for name in trainable if not any(token in name for token in allowed)]
            if unexpected:
                raise RuntimeError(
                    "Linear probing exposed parameters outside the fixed-vocabulary classification readout:\n"
                    + "\n".join(unexpected)
                )
            expected_scales = 6 if getattr(self.model.module if hasattr(self.model, "module") else self.model, "end2end", False) else 3
            expected_tensors = expected_scales * 2
            if len(trainable) != expected_tensors:
                raise RuntimeError(
                    f"Expected {expected_tensors} trainable classifier tensors, but found {len(trainable)}: {trainable}"
                )
            LOGGER.info("Linear probing trainable parameter whitelist:\n" + "\n".join(trainable))


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """Trainer for YOLOE segmentation models trained from scratch without pretrained weights."""

    pass


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """Trainer for YOLOE segmentation models with Vision Prompt (VP) capabilities."""

    pass
