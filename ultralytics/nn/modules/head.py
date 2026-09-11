# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN, \
    SNNContrastiveHead
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
from .text_guided_ser import SafeTextGuidedBiLevelRoutingAttentionSNN, TextGuidedBiLevelRoutingAttentionSNN
from .yolo_spikformer import MS_ConvBlock, MS_StandardConv, SpikeConv, SpikeConvWithoutBN

__all__ = (
    "OBB",
    "Classify",
    "Detect",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "YOLOEDetect",
    "YOLOESNNDetect",
    "YOLOESNNDetect2",
    "YOLOESNNDetect2_1",
    "YOLOESNNDetect2TextGuidedSER",
    "YOLOESNNSegment20",
    "YOLOESNNSegment21",
    "YOLOESNNSegment22",
    "YOLOESNNSegment22_1",
    "YOLOESNNSegment22_2",
    "YOLOESNNSegment22_2ResidualTextSER",
    "YOLOESNNSegment22_2SafeTextRouteSER",
    "YOLOESNNSegment22_2SharedTextRouteSER",
    "omnispike_SNNSegment22_2DirectTextRouteSER",
    "YOLOESNNSegment2TextGuidedSER",
    "YOLOESNNDetect3",
    "YOLOESegment",
    "v10Detect",
)


class TemporalAggregate(nn.Module):
    """Collapse the SNN time dimension before the final 2D projection layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文说明：
        # SNN 主干输出的是 [T, B, C, H, W] 五维时序特征，
        # 但 YOLOE 原有的分类/回归头和对比学习头都按 [B, C, H, W] 设计。
        # 因此这里统一对时间维做平均，把时序特征压回普通 2D 特征。
        return x.mean(0) if x.ndim == 5 else x


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v5/v5/v8/v9/11 backward compatibility."""
        return hasattr(self, "one2one")

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]  # TODO: need to test obb
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x).
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x, y, w, h, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x, y, w, h, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        # 中文说明：
        # YOLOE 与普通 Detect 最大的区别在于：
        # 1. cv2 仍然负责框回归
        # 2. cv3 不直接输出类别分数，而是输出图像语义 embedding
        # 3. cv4 再把图像 embedding 和文本 embedding 做对比学习匹配，得到分类分数
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        # 中文说明：
        # cv4 是真正的“图文对齐头”。
        # 输入：
        #   - 来自 cv3 的图像 embedding，形状通常为 [B, embed, H, W]
        #   - 文本提示 embedding，形状通常为 [B, num_classes, embed]
        # 输出：
        #   - 每个位置对每个类别的匹配分数
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            # 中文说明：
            # end2end 模式下保留 one2many / one2one 两套分支，
            # 这和当前 YOLO26 / YOLOE 的训练损失设计一致。
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        # 中文说明：
        # reprta：文本提示的小型适配模块。
        # 作用是把文本编码器输出的原始文本特征再调整一下，
        # 变成更适合检测头对齐的文本 embedding。
        self.reprta = Residual(SwiGLUFFN(embed, embed))
        # 中文说明：
        # savpe：视觉提示编码模块。
        # 当用户给的是 visual prompt 而不是文本时，
        # 用它把视觉提示区域编码成和文本 embedding 同维度的提示向量。
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        # 中文说明：
        # fuse() 用于把“文本提示”融合进检测头最后的 1x1 卷积里。
        # 融合后推理时就不需要每次再显式传文本 embedding 参与匹配，
        # 相当于把开放词汇头临时重参数化成一个固定词表检测头。
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        # 中文说明：
        # 对每个检测尺度，把文本 embedding 与分类头最后一层卷积权重融合，
        # 这样原本的“图像 embedding × 文本 embedding”匹配，
        # 可以转成一个普通的 1x1 卷积输出。
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        # 中文说明：
        # 文本提示先经过 reprta 做适配，再做 L2 归一化。
        # 归一化后更适合与图像 embedding 做余弦/相似度匹配。
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    @staticmethod
    def _aggregate_branch_output(y: torch.Tensor) -> torch.Tensor:
        """Aggregate one branch output from [T, B, C, H, W] to [B, C, H, W] when needed."""
        return y.mean(0) if y.ndim == 5 else y

    @staticmethod
    def _branch_bias(module: nn.Module) -> torch.Tensor:
        """Return the writable bias tensor from either SpikeConvWithoutBN or fused Conv2d layers."""
        if isinstance(module, SpikeConvWithoutBN):
            return module.conv.bias
        if isinstance(module, nn.Conv2d):
            return module.bias
        raise TypeError(f"Unsupported branch output layer type: {type(module).__name__}")

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        # 中文说明：
        # 如果传入的是视觉提示：
        # 1. 先结合多尺度特征和提示区域
        # 2. 编码成 [B, N, embed] 的提示向量
        # 3. 后续和文本提示一样进入统一的 class embedding 流程
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        # 中文说明：
        # prompt-free 模式下，直接走已经 fuse 完成的 LRPC 分支。
        # 否则继续使用标准 YOLOE 的开放词汇 forward。
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        # 中文说明：
        # 这里是 fuse 后的 prompt-free 推理分支：
        # 不再显式计算“图像 embedding 与文本 embedding 的相似度”，
        # 而是直接使用融合好的轻量分类头和定位头。
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv2
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompts into SpikeYOLO-style SNN cls branches for prompt-free inference."""
        # 中文说明：
        # 原始 YOLOE 默认 cls 分支最后一层是 Conv2d；
        # 这里最后一层是 SpikeConvWithoutBN，需要先取出内部 conv 再与 BNContrastiveHead 融合。
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv_module = cls_h[-1]
            assert isinstance(conv_module, SpikeConvWithoutBN)
            conv = conv_module.conv
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            fused_conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )
            fused_conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            fused_conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = fused_conv
            bn_h.fuse()

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and text embeddings."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        # 中文说明：
        # boxes：仍然是标准检测头逻辑，来自 cv2 的回归输出
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        # 中文说明：
        # x[-1] 是外部在 YOLOEModel.predict() 里 append 进来的 class embedding，
        # 它可能来自：
        # 1. 文本提示 text prompt
        # 2. 视觉提示 visual prompt
        # 3. 二者拼接
        self.nc = x[-1].shape[1]
        # 中文说明：
        # scores 的生成不是普通分类卷积，而是：
        # 图像 embedding（cv3 输出） + class embedding（x[-1]） -> contrastive_head -> 分类分数
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # 中文说明：
        # box_head 的 bias 仍按检测头方式初始化；
        # cls_head 的最后一层先清零；
        # contrastive_head 的 bias 按类别先验做初始化。
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )
            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv
            bn_h.fuse()


class YOLOESNNDetect(YOLOEDetect):
    """YOLOE 的 SNN 检测头实现。"""

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        # 当前版本参考 SpikeYOLO 的检测头设计：
        # 1. box 分支和 image embedding 分支都先在 5D 时序特征上完成卷积；
        # 2. box 分支末端继续使用 SpikeConvWithoutBN，保持 SNN 风格输出；
        # 3. 语义分支在末端先做时间聚合，再用普通 1x1 Conv2d 投影到 embedding 空间；
        # 4. 在 head 内部再把结果接回 YOLOE 原本的
        #     prompt、contrastive head、loss 和推理协议。
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c2, 3),
                SpikeConv(c2, c2, 3),
                SpikeConvWithoutBN(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c3, 3),
                SpikeConv(c3, c3, 3),
                SpikeConvWithoutBN(c3, embed, 1),
            )
            for x in ch
        )
        # 这里单独补一个 2D 语义适配层：
        # SpikeYOLO 风格的 spike 分支先完整输出，再在时序聚合后映射到更稳定的连续语义空间，
        # 供 YOLOE 的 contrastive head 做图文对齐。
        self.cls_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(embed, embed, 1), nn.GroupNorm(32, embed)) for _ in ch)
        if end2end:
            # end2end 模式下仍然保留 YOLOE 的 one2many / one2one 双分支结构。
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cls_proj = copy.deepcopy(self.cls_proj)

    @staticmethod
    def _to_2d_feats(x: list[torch.Tensor]) -> list[torch.Tensor]:
        """把 [T, B, C, H, W] 特征聚合成普通 BCHW 特征。"""
        return [xi.mean(0) if xi.ndim == 5 else xi for xi in x]

    @staticmethod
    def _aggregate_branch_output(y: torch.Tensor) -> torch.Tensor:
        """把单个分支输出从 5D 聚合回 2D。"""
        return y.mean(0) if y.ndim == 5 else y

    @staticmethod
    def _branch_bias(module: nn.Module) -> torch.Tensor:
        """返回分支最后一层可写的 bias 张量。"""
        if isinstance(module, SpikeConvWithoutBN):
            return module.conv.bias
        if isinstance(module, nn.Conv2d):
            return module.bias
        raise TypeError(f"Unsupported branch output layer type: {type(module).__name__}")

    @staticmethod
    def _normalize_cls_feature(cls_feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """把 spike 语义分支输出映射到更稳定的 2D 连续语义空间。"""
        return F.normalize(proj(cls_feat), dim=1)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        # SAVPE 仍然是 2D 模块，因此这里先把时序特征聚合成 BCHW，
        # 再复用父类的 visual prompt 编码逻辑。
        return super().get_vpe(self._to_2d_feats(x), vpe)

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """把文本提示融合进分类分支，用于 prompt-free 推理。"""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv_module = cls_h[-1]
            if isinstance(conv_module, SpikeConvWithoutBN):
                conv = conv_module.conv
            else:
                assert isinstance(conv_module, nn.Conv2d)
                conv = conv_module
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            fused_conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )
            fused_conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            fused_conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = fused_conv
            bn_h.fuse()

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """基于时序分支输出构建 YOLOE 的 box 和 score 结果。"""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:
            return dict()

        bs = x[0].shape[1] if x[0].ndim == 5 else x[0].shape[0]
        feats_2d = self._to_2d_feats(x[:3])

        # box 分支保持 SpikeYOLO 的纯 SNN 输出形式；
        # cls 分支也先按 SpikeYOLO 的方式得到 spike 特征，再额外经过 2D 语义适配层与文本特征对齐。
        boxes = torch.cat(
            [self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(self._aggregate_branch_output(cls_head[i](x[i])), self.cls_proj[i]),
                    x[-1],
                ).reshape(
                    bs, self.nc, -1
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.no = self.nc + self.reg_max * 4
        return dict(boxes=boxes, scores=scores, feats=feats_2d)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        # 对外接口保持和 YOLOEDetect 一致，只是前三个输入允许是 5D 时序特征。
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])

        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1]]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """融合文本提示后的 prompt-free 推理分支。"""
        boxes, scores, index = [], [], []
        feats_2d = self._to_2d_feats(x)
        bs = feats_2d[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cls_proj = self.cls_proj if not self.end2end else self.one2one_cls_proj

        for i in range(self.nl):
            loc_feat = self._aggregate_branch_output(cv2[i](x[i]))
            cls_feat = self._normalize_cls_feature(self._aggregate_branch_output(cv3[i](x[i])), cls_proj[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=feats_2d, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """初始化检测头偏置，并兼容 SpikeConvWithoutBN 作为输出层。"""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            self._branch_bias(a[-1]).data[:] = 2.0
            self._branch_bias(b[-1]).data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                self._branch_bias(a[-1]).data[:] = 2.0
                self._branch_bias(b[-1]).data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESNNDetect2(Detect):
    """YOLOE-compatible SNN detect head implemented directly on top of Detect."""

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        super().__init__(nc, reg_max, end2end, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c2, 3),
                SpikeConv(c2, c2, 3),
                SpikeConvWithoutBN(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c3, 3),
                SpikeConv(c3, c3, 3),
                SpikeConvWithoutBN(c3, embed, 1),
            )
            for x in ch
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        # self.cv4 = nn.ModuleList(SNNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.cls_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(embed, embed, 1), nn.GroupNorm(32, embed)) for _ in ch)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cls_proj = copy.deepcopy(self.cls_proj)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    @staticmethod
    def _to_2d_feats(x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [xi.mean(0) if xi.ndim == 5 else xi for xi in x]

    @staticmethod
    def _aggregate_branch_output(y: torch.Tensor) -> torch.Tensor:
        return y.mean(0) if y.ndim == 5 else y

    @staticmethod
    def _branch_bias(module: nn.Module) -> torch.Tensor:
        if isinstance(module, SpikeConvWithoutBN):
            return module.conv.bias
        if isinstance(module, nn.Conv2d):
            return module.bias
        raise TypeError(f"Unsupported branch output layer type: {type(module).__name__}")

    @staticmethod
    def _normalize_cls_feature(cls_feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        return F.normalize(proj(cls_feat), dim=1, eps=1e-6)

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        if txt_feats is None:
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv_module = cls_h[-1]
            if isinstance(conv_module, SpikeConvWithoutBN):
                conv = conv_module.conv
            else:
                assert isinstance(conv_module, nn.Conv2d)
                conv = conv_module

            t = txt_feats * bn_h.logit_scale.exp()
            conv = fuse_conv_and_bn(conv, bn_h.norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data
            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bn_h.bias

            fused_conv = nn.Conv2d(conv.in_channels, w.shape[0], kernel_size=1).requires_grad_(False).to(conv.weight.device)
            fused_conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            fused_conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = fused_conv
            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        x = self._to_2d_feats(x)
        if vpe.shape[1] == 0:
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:
            return dict()

        bs = x[0].shape[1] if x[0].ndim == 5 else x[0].shape[0]
        feats_2d = self._to_2d_feats(x[:3])
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj

        # 正常检测路径：
        # 1. x[: երեք个多尺度特征，x[-1] 是 tasks.py 在进入 head 前 append 进来的 class/text embedding；
        # 2. box_head 也就是 cv2，负责输出回归分支；
        # 3. cls_head 也就是 cv3，先输出图像语义 embedding；
        # 4. contrastive_head 也就是 cv4，再把“图像 embedding + 文本 embedding”匹配成分类分数。
        #
        # 和普通 YOLO Detect 最大区别：
        # 普通 Detect 是 cv3 直接输出类别 logits；
        # YOLOE/YOLOESNNDetect2 是 cv3 先产出 image embedding，真正的分类由 cv4 完成。
        boxes = torch.cat(
            [self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        # 这里的执行顺序是：
        # P3/P4/P5 特征 -> cv3 -> 时间维聚合 -> cls_proj -> 归一化 -> cv4(与文本 embedding 做匹配) -> scores
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(self._aggregate_branch_output(cls_head[i](x[i])), cls_proj[i]),
                    x[-1],
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.no = self.nc + self.reg_max * 4
        return dict(boxes=boxes, scores=scores, feats=feats_2d)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])

        # 正常检测训练/推理路径：
        # 1. 先走 one2many 分支，得到 box 回归结果和分类分数；
        # 2. 如果开启 end2end，再额外构造 one2one 分支给 E2E loss / 后处理使用；
        # 3. 训练态直接返回 preds 给 loss，不在 head 里做 decode；
        # 4. 推理态才进入 _inference()，把 boxes 解码成真实框。
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1]]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            # 训练时返回的不是最终 NMS 后框，
            # 而是 loss 需要的中间结果：boxes / scores / feats。
            return preds

        # 推理时才把分布式回归结果 decode 成实际框坐标，
        # 并把分类分数 sigmoid 后拼接成最终预测。
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        boxes, scores, index = [], [], []
        feats_2d = self._to_2d_feats(x)
        bs = feats_2d[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cls_proj = self.cls_proj if not self.end2end else self.one2one_cls_proj

        for i in range(self.nl):
            loc_feat = self._aggregate_branch_output(cv2[i](x[i]))
            cls_feat = self._normalize_cls_feature(self._aggregate_branch_output(cv3[i](x[i])), cls_proj[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=feats_2d, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            self._branch_bias(a[-1]).data[:] = 2.0
            self._branch_bias(b[-1]).data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                self._branch_bias(a[-1]).data[:] = 2.0
                self._branch_bias(b[-1]).data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

class SNNClassificationProjection(nn.Module):
    """Use a LIF spike activation before the learnable 1x1 visual-semantic projection."""

    def __init__(self, embed: int):
        super().__init__()
        self.spike_conv = SpikeConvWithoutBN(embed, embed, 1)
        self.norm = nn.GroupNorm(32, embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SNNClassificationProjection expects [T,B,C,H,W], got {tuple(x.shape)}")
        # 与 YOLOESNNDetect2 不同：先保留时间维并通过 LIF 发放，再让 1x1 卷积接收脉冲输入。
        x = self.spike_conv(x)
        # 1x1 脉冲卷积完成后才聚合时间维；后续归一化和图文匹配仍使用连续读出。
        return self.norm(x.mean(0))


class YOLOESNNDetect2_1(Detect):
    """YOLOESNNDetect2 variant with spike-driven cls_proj convolutions."""

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        super().__init__(nc, reg_max, end2end, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c2, 3),
                SpikeConv(c2, c2, 3),
                SpikeConvWithoutBN(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(x, c3, 3),
                SpikeConv(c3, c3, 3),
                SpikeConvWithoutBN(c3, embed, 1),
            )
            for x in ch
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        # self.cv4 = nn.ModuleList(SNNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.cls_proj = nn.ModuleList(SNNClassificationProjection(embed) for _ in ch)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cls_proj = copy.deepcopy(self.cls_proj)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    @staticmethod
    def _to_2d_feats(x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [xi.mean(0) if xi.ndim == 5 else xi for xi in x]

    @staticmethod
    def _aggregate_branch_output(y: torch.Tensor) -> torch.Tensor:
        return y.mean(0) if y.ndim == 5 else y

    @staticmethod
    def _branch_bias(module: nn.Module) -> torch.Tensor:
        if isinstance(module, SpikeConvWithoutBN):
            return module.conv.bias
        if isinstance(module, nn.Conv2d):
            return module.bias
        raise TypeError(f"Unsupported branch output layer type: {type(module).__name__}")

    @staticmethod
    def _normalize_cls_feature(cls_feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        return F.normalize(proj(cls_feat), dim=1, eps=1e-6)

    @staticmethod
    def _build_coco_transfer_classifier(txt_feats: torch.Tensor, contrastive_head: nn.Module) -> nn.Conv2d:
        """Build the fixed-vocabulary linear classifier used by YOLOE COCO transfer.

        The stock YOLOE head can fuse text directly into the last cv3 convolution. This SNN head has an extra
        spike-driven cls_proj between cv3 and the contrastive head, so fusing into cv3 would produce an invalid
        nc-channel tensor for a projection that still expects ``embed`` channels. Instead, fuse the frozen COCO text
        vocabulary and BNContrastiveHead into an equivalent 1x1 classifier *after* cls_proj.
        """
        if not isinstance(contrastive_head, BNContrastiveHead):
            raise TypeError(
                "COCO transfer currently requires BNContrastiveHead, "
                f"but received {type(contrastive_head).__name__}."
            )
        txt_feats = F.normalize(txt_feats.to(torch.float32), dim=-1, p=2)
        norm = contrastive_head.norm
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        shift = norm.bias - norm.running_mean * scale
        logit_scale = contrastive_head.logit_scale.clamp(max=math.log(50.0)).exp()

        weight = logit_scale * txt_feats * scale.unsqueeze(0)
        bias = logit_scale * (txt_feats @ shift) + contrastive_head.bias
        classifier = nn.Conv2d(txt_feats.shape[-1], txt_feats.shape[0], 1).to(
            device=norm.weight.device, dtype=norm.weight.dtype
        )
        classifier.weight.data.copy_(weight.to(classifier.weight.dtype).unsqueeze(-1).unsqueeze(-1))
        classifier.bias.data.copy_(bias.to(classifier.bias.dtype))
        return classifier

    @torch.no_grad()
    def prepare_coco_transfer(self, txt_feats: torch.Tensor) -> None:
        """Reparameterize the open-vocabulary similarity readout for closed-set COCO transfer."""
        if hasattr(self, "transfer_cls"):
            return
        txt_feats = txt_feats.to(next(self.parameters()).device, dtype=torch.float32).squeeze(0)
        self.transfer_cls = nn.ModuleList(
            self._build_coco_transfer_classifier(txt_feats, contrastive_head) for contrastive_head in self.cv4
        )
        if self.end2end:
            self.one2one_transfer_cls = nn.ModuleList(
                self._build_coco_transfer_classifier(txt_feats, contrastive_head)
                for contrastive_head in self.one2one_cv4
            )
        self.nc = txt_feats.shape[0]

    def _classification_logits(
        self,
        cls_feat: torch.Tensor,
        text_feats: torch.Tensor,
        cls_head: nn.Module,
        contrastive_head: nn.Module,
        scale_index: int,
    ) -> torch.Tensor:
        """Project a classification feature and apply either open-vocabulary or COCO-transfer readout."""
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj
        projected = self._normalize_cls_feature(cls_feat, cls_proj[scale_index])
        if hasattr(self, "transfer_cls"):
            transfer_head = self.transfer_cls if cls_head is self.cv3 else self.one2one_transfer_cls
            return transfer_head[scale_index](projected)
        return contrastive_head(projected, text_feats)

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        if txt_feats is None:
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv_module = cls_h[-1]
            if isinstance(conv_module, SpikeConvWithoutBN):
                conv = conv_module.conv
            else:
                assert isinstance(conv_module, nn.Conv2d)
                conv = conv_module

            t = txt_feats * bn_h.logit_scale.exp()
            conv = fuse_conv_and_bn(conv, bn_h.norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data
            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bn_h.bias

            fused_conv = nn.Conv2d(conv.in_channels, w.shape[0], kernel_size=1).requires_grad_(False).to(conv.weight.device)
            fused_conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            fused_conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = fused_conv
            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        x = self._to_2d_feats(x)
        if vpe.shape[1] == 0:
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:
            return dict()

        bs = x[0].shape[1] if x[0].ndim == 5 else x[0].shape[0]
        feats_2d = self._to_2d_feats(x[:3])
        # 正常检测路径：
        # 1. x[: երեք个多尺度特征，x[-1] 是 tasks.py 在进入 head 前 append 进来的 class/text embedding；
        # 2. box_head 也就是 cv2，负责输出回归分支；
        # 3. cls_head 也就是 cv3，先输出图像语义 embedding；
        # 4. contrastive_head 也就是 cv4，再把“图像 embedding + 文本 embedding”匹配成分类分数。
        #
        # 和普通 YOLO Detect 最大区别：
        # 普通 Detect 是 cv3 直接输出类别 logits；
        # YOLOE/YOLOESNNDetect2 是 cv3 先产出 image embedding，真正的分类由 cv4 完成。
        boxes = torch.cat(
            [self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        # 这里的执行顺序是：
        # 与 YOLOESNNDetect2 不同：P3/P4/P5 -> cv3 -> LIF -> cls_proj 1x1卷积
        # -> 时间维聚合 -> GroupNorm/L2归一化 -> cv4。
        scores = torch.cat(
            [
                self._classification_logits(cls_head[i](x[i]), x[-1], cls_head, contrastive_head[i], i).reshape(
                    bs, self.nc, -1
                )
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.no = self.nc + self.reg_max * 4
        return dict(boxes=boxes, scores=scores, feats=feats_2d)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])

        # 正常检测训练/推理路径：
        # 1. 先走 one2many 分支，得到 box 回归结果和分类分数；
        # 2. 如果开启 end2end，再额外构造 one2one 分支给 E2E loss / 后处理使用；
        # 3. 训练态直接返回 preds 给 loss，不在 head 里做 decode；
        # 4. 推理态才进入 _inference()，把 boxes 解码成真实框。
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1]]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            # 训练时返回的不是最终 NMS 后框，
            # 而是 loss 需要的中间结果：boxes / scores / feats。
            return preds

        # 推理时才把分布式回归结果 decode 成实际框坐标，
        # 并把分类分数 sigmoid 后拼接成最终预测。
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        boxes, scores, index = [], [], []
        feats_2d = self._to_2d_feats(x)
        bs = feats_2d[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cls_proj = self.cls_proj if not self.end2end else self.one2one_cls_proj

        for i in range(self.nl):
            loc_feat = self._aggregate_branch_output(cv2[i](x[i]))
            # 与 YOLOESNNDetect2 不同：prompt-free 路径同样先进行脉冲投影，再聚合为连续 embedding。
            cls_feat = self._normalize_cls_feature(cv3[i](x[i]), cls_proj[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=feats_2d, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            self._branch_bias(a[-1]).data[:] = 2.0
            self._branch_bias(b[-1]).data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                self._branch_bias(a[-1]).data[:] = 2.0
                self._branch_bias(b[-1]).data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

class SNNMaskCoefficientHead(nn.Module):
    """Use spike-driven convolutions to extract mask features and a continuous 1x1 convolution for readout."""

    def __init__(self, c1: int, c_hidden: int, nm: int):
        super().__init__()
        self.cv1 = MS_StandardConv(c1, c_hidden, 3)
        self.cv2 = MS_StandardConv(c_hidden, c_hidden, 3)
        # 连续读出层不再脉冲化：mask 系数需要保留正负值，才能自由组合共享原型。
        self.readout = nn.Conv2d(c_hidden, nm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SNN mask coefficient head expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.cv2(self.cv1(x))
        return self.readout(x.mean(0))


class SNNMaskProto(nn.Module):
    """Generate mask prototypes with spike-driven feature extraction and a continuous output projection."""

    def __init__(self, c1: int, c_hidden: int, nm: int):
        super().__init__()
        self.cv1 = MS_StandardConv(c1, c_hidden, 3)
        # 只放大空间维，时间维 T 保持不变，继续维持 [T,B,C,H,W] 的 SNN 张量协议。
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.cv2 = MS_StandardConv(c_hidden, c_hidden, 3)
        # 与 2.1 相同，最终输出保留连续读出，避免把 prototype logits 限制为非负整数脉冲。
        self.readout = nn.Conv2d(c_hidden, nm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SNN mask prototype head expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.cv2(self.upsample(self.cv1(x)))
        return self.readout(x.mean(0))


class SNNMaskCoefficientHeadSpikeReadout(nn.Module):
    """Generate continuous mask coefficients with a spike-driven final 1x1 convolution."""

    def __init__(self, c1: int, c_hidden: int, nm: int):
        super().__init__()
        self.cv1 = MS_StandardConv(c1, c_hidden, 3)
        self.cv2 = MS_StandardConv(c_hidden, c_hidden, 3)
        self.readout = SpikeConvWithoutBN(c_hidden, nm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SNN mask coefficient head expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.cv2(self.cv1(x))
        # 与 22_1 不同：readout 内先通过 LIF 发放，1x1 卷积接收脉冲，卷积后才聚合时间维。
        return self.readout(x).mean(0)


class SNNMaskProtoSpikeReadout(nn.Module):
    """Generate continuous mask prototypes with a spike-driven final 1x1 convolution."""

    def __init__(self, c1: int, c_hidden: int, nm: int):
        super().__init__()
        self.cv1 = MS_StandardConv(c1, c_hidden, 3)
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.cv2 = MS_StandardConv(c_hidden, c_hidden, 3)
        self.readout = SpikeConvWithoutBN(c_hidden, nm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SNN mask prototype head expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.cv2(self.upsample(self.cv1(x)))
        # 与 22_1 不同：保留时间维完成脉冲1x1卷积，再平均为连续的 prototype logits。
        return self.readout(x).mean(0)


# SNN 分割头版本演进说明：
# 1. Segment20：检测部分使用 SNN，新增的 mask coefficient（cv5）和 Proto 分支均使用标准 ANN。
# 2. Segment21：只把 cv5 的特征提取改为 SNN，cv5 最终 1x1 仍为连续 ANN 读出，Proto 仍为 ANN。
# 3. Segment22：在 Segment21 的思想上继续把 Proto 特征提取改为 SNN；cv5 和 Proto 最终均连续读出。
# 4. Segment22_1：分割结构与 Segment22 相同，但继承 YOLOESNNDetect2_1，使开放词汇 cls_proj 接收脉冲输入。
# 5. Segment22_2：在 Segment22_1 上进一步让 cv5 和 Proto 的最终 1x1 卷积也接收 LIF 脉冲输入；
#    两个分支仍在卷积后聚合时间维，输出连续 mask coefficient 和连续 Proto，供最终掩码线性组合使用。
class YOLOESNNSegment20(YOLOESNNDetect2):
    """SNN 分割 2.0：SNN 检测头加标准 ANN mask coefficient 与 ANN Proto，作为分割基线。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        # 2.0 基线新增部分：沿用 Ultralytics 标准 ANN Proto，先聚合时间维再生成共享 mask 原型。
        self.proto = Proto(ch[0], self.npr, self.nm)
        c5 = max(ch[0] // 4, self.nm)
        # 2.0 基线新增部分：三个尺度均使用普通 Conv2d/BN/SiLU 预测连续 mask 系数。
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(channels, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for channels in ch
        )
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_mask_coefficients(self, x: list[torch.Tensor], mask_head: nn.Module) -> torch.Tensor:
        bs = x[0].shape[1] if x[0].ndim == 5 else x[0].shape[0]
        feats_2d = self._to_2d_feats(x[:3])
        return torch.cat(
            [mask_head[i](feats_2d[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )

    def _forward_proto(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.proto(self._to_2d_feats(x[:3])[0])

    def forward_head(self, x, box_head, cls_head, mask_head=None, contrastive_head=None):
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self._forward_proto(x)
        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None


class YOLOESNNSegment21(YOLOESNNDetect2):
    """SNN 分割 2.1：cv5 使用 SNN 提取特征并连续读出，Proto 分支仍使用标准 ANN。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        # 相比 2.0：Proto 保持标准 ANN 实现，仍在时间聚合后的 P3 特征上生成共享 mask 原型。
        self.proto = Proto(ch[0], self.npr, self.nm)
        c5 = max(ch[0] // 4, self.nm)
        # 相比 2.0 的唯一结构变化：cv5 的两层 ANN Conv 改为 MS_StandardConv，末端仍连续读出。
        self.cv5 = nn.ModuleList(SNNMaskCoefficientHead(channels, c5, self.nm) for channels in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_mask_coefficients(self, x: list[torch.Tensor], mask_head: nn.Module) -> torch.Tensor:
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("YOLOESNNSegment21 expects P3/P4/P5 in [T,B,C,H,W] format")
        bs = x[0].shape[1]
        return torch.cat(
            [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )


    def _forward_proto(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.proto(self._to_2d_feats(x[:3])[0])

    def forward_head(self, x, box_head, cls_head, mask_head=None, contrastive_head=None):
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self._forward_proto(x)
        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None


class YOLOESNNSegment22(YOLOESNNDetect2):
    """SNN 分割 2.2：cv5 与 Proto 均使用 SNN 提取特征，最终 1x1 卷积仍是连续 ANN 读出。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        c5 = max(ch[0] // 4, self.nm)
        # 与 2.1 相同：mask 系数分支使用 SNN 特征提取，最后保留连续 1x1 读出。
        self.cv5 = nn.ModuleList(SNNMaskCoefficientHead(channels, c5, self.nm) for channels in ch)
        # 相比 2.1 的唯一结构变化：ANN Proto 改为 SNN prototype 特征提取，最终 1x1 连续读出保留。
        self.proto = SNNMaskProto(ch[0], self.npr, self.nm)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_mask_coefficients(self, x: list[torch.Tensor], mask_head: nn.Module) -> torch.Tensor:
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("YOLOESNNSegment22 expects P3/P4/P5 in [T,B,C,H,W] format")
        bs = x[0].shape[1]
        return torch.cat(
            [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )

    def _forward_proto(self, x: list[torch.Tensor]) -> torch.Tensor:
        if x[0].ndim != 5:
            raise ValueError("YOLOESNNSegment22 expects P3 in [T,B,C,H,W] format")
        return self.proto(x[0])

    def forward_head(self, x, box_head, cls_head, mask_head=None, contrastive_head=None):
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self._forward_proto(x)
        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None

class YOLOESNNSegment22_1(YOLOESNNDetect2_1):
    """SNN 分割 2.2.1：分割分支同 Segment22，并将开放词汇 cls_proj 改为脉冲输入投影。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        c5 = max(ch[0] // 4, self.nm)
        # 与 2.1 相同：mask 系数分支使用 SNN 特征提取，最后保留连续 1x1 读出。
        self.cv5 = nn.ModuleList(SNNMaskCoefficientHead(channels, c5, self.nm) for channels in ch)
        # 相比 2.1 的唯一结构变化：ANN Proto 改为 SNN prototype 特征提取，最终 1x1 连续读出保留。
        self.proto = SNNMaskProto(ch[0], self.npr, self.nm)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_mask_coefficients(self, x: list[torch.Tensor], mask_head: nn.Module) -> torch.Tensor:
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("YOLOESNNSegment22 expects P3/P4/P5 in [T,B,C,H,W] format")
        bs = x[0].shape[1]
        return torch.cat(
            [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )

    def _forward_proto(self, x: list[torch.Tensor]) -> torch.Tensor:
        if x[0].ndim != 5:
            raise ValueError("YOLOESNNSegment22 expects P3 in [T,B,C,H,W] format")
        return self.proto(x[0])

    def forward_head(self, x, box_head, cls_head, mask_head=None, contrastive_head=None):
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self._forward_proto(x)
        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None


class YOLOESNNSegment22_2(YOLOESNNDetect2_1):
    """SNN 分割 2.2.2：cls_proj、mask coefficient 读出和 Proto 读出的 1x1 卷积均接收脉冲输入。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        c5 = max(ch[0] // 4, self.nm)
        # 与 22_1 不同：三个 mask coefficient 的最终1x1卷积都在输入前增加 LIF 发放。
        self.cv5 = nn.ModuleList(SNNMaskCoefficientHeadSpikeReadout(channels, c5, self.nm) for channels in ch)
        # 与 22_1 不同：Proto 最终1x1卷积同样改为脉冲输入，但输出仍保留连续值。
        self.proto = SNNMaskProtoSpikeReadout(ch[0], self.npr, self.nm)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_mask_coefficients(self, x: list[torch.Tensor], mask_head: nn.Module) -> torch.Tensor:
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("YOLOESNNSegment22 expects P3/P4/P5 in [T,B,C,H,W] format")
        bs = x[0].shape[1]
        return torch.cat(
            [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            dim=2,
        )

    def _forward_proto(self, x: list[torch.Tensor]) -> torch.Tensor:
        if x[0].ndim != 5:
            raise ValueError("YOLOESNNSegment22 expects P3 in [T,B,C,H,W] format")
        return self.proto(x[0])

    def forward_head(self, x, box_head, cls_head, mask_head=None, contrastive_head=None):
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self._forward_proto(x)
        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None


class YOLOESNNSegment22_2ResidualTextSER(YOLOESNNSegment22_2):
    """Segment22_2 with weak residual text-guided SER on the P5 classification path only."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 2,
        topk_text: int = 8,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, nm, npr, embed, with_bn, reg_max, end2end, ch)
        # 与 Segment22_2 不同：只在语义最强的 P5 分类路径增加 Text-SER，P3/P4 完全保持基线。
        self.p5_text_ser = TextGuidedBiLevelRoutingAttentionSNN(
            dim=ch[2],
            text_dim=embed,
            num_heads=ser_num_heads,
            n_win=ser_n_win,
            topk=ser_topk,
            topk_text=topk_text,
            # β 固定为 0.01：文本只弱引导稀疏区域路由，避免训练词表主导视觉拓扑。
            fixed_text_strength=0.01,
            # 文本已经参与路由，不再把文本向量作为连续残差直接写进视觉特征。
            use_text_residual=False,
            # β同时连续调节已聚合区域的响应，使文本投影可训练，避免梯度仅经过不可导 Top-K。
            use_text_attention_gate=True,
        )
        # α=0.1*sigmoid(logit)，初始约 0.001 且始终限制在 (0, 0.1)。
        self.p5_ser_alpha_logit = nn.Parameter(torch.tensor(-4.59511985))

    @property
    def p5_ser_alpha(self) -> torch.Tensor:
        return 0.1 * torch.sigmoid(self.p5_ser_alpha_logit)

    def _forward_head_with_guided_p5(
        self, x, guided_p5, box_head, cls_head, mask_head=None, contrastive_head=None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        if len(x) != 4 or any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Residual Text-SER segment head expects P3/P4/P5 plus text embeddings")

        bs = x[0].shape[1]
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj
        # 与 Segment22_2 相同：框回归和用于 anchor 的 feats 始终使用未经注意力修改的原始特征。
        boxes = torch.cat(
            [self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        cls_inputs = [x[0], x[1], guided_p5]
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(cls_head[i](cls_inputs[i]), cls_proj[i]),
                    x[-1],
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        preds = dict(boxes=boxes, scores=scores, feats=self._to_2d_feats(x[:3]))
        if mask_head is not None:
            # 与 Segment22_2 相同：mask coefficient 使用原始 P3/P4/P5，不接收 Text-SER 特征。
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        if len(x) != 4:
            raise ValueError(f"Expected P3/P4/P5 plus text embeddings, got {len(x)} inputs")

        # 文本弱引导 P5 的稀疏区域聚合；小 α 残差保证初始化时几乎等价于 Segment22_2。
        p5_delta = self.p5_text_ser(x[2], x[-1])
        guided_p5 = x[2] + self.p5_ser_alpha * p5_delta
        preds = self._forward_head_with_guided_p5(x, guided_p5, **self.one2many)

        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1]]
            # 与基线 one-to-one 语义相同：阻断共享视觉/SER特征的梯度，只训练 one-to-one 读出头。
            one2one = self._forward_head_with_guided_p5(x_detach, guided_p5.detach(), **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        proto = self._forward_proto(x)
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = proto.detach()
        else:
            preds["proto"] = proto
        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        outputs = y if self.export else (y, preds)
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class YOLOESNNSegment22_2SafeTextRouteSER(YOLOESNNSegment22_2):
    """Segment22_2 with safe text-routed P4/P5 attention and spike-convolution refinement."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, nm, npr, embed, with_bn, reg_max, end2end, ch)
        route_kwargs = dict(
            text_dim=embed,
            num_heads=ser_num_heads,
            n_win=ser_n_win,
            topk=ser_topk,
            candidate_pool=candidate_pool,
            visual_keep=visual_keep,
        )
        # 与0725基线不同：P4、P5分类路径都用文本在视觉候选池内选路，P3和定位/掩码路径保持原样。
        self.p4_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=ch[1], **route_kwargs)
        self.p5_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=ch[2], **route_kwargs)
        # 参考0603：注意力结果不直接进入分类读出，而是先经过SNN卷积块重新编码。
        self.p4_ser_refine = MS_ConvBlock(ch[1], mlp_ratio=1, sep_kernel_size=7)
        self.p5_ser_refine = MS_ConvBlock(ch[2], mlp_ratio=1, sep_kernel_size=7)

        self.ser_residual_scale_max = 0.05
        initial_scale = 0.005
        initial_ratio = initial_scale / self.ser_residual_scale_max
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.p4_ser_scale_logit = nn.Parameter(torch.tensor(initial_logit))
        self.p5_ser_scale_logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def p4_ser_scale(self) -> torch.Tensor:
        return self.ser_residual_scale_max * torch.sigmoid(self.p4_ser_scale_logit)

    @property
    def p5_ser_scale(self) -> torch.Tensor:
        return self.ser_residual_scale_max * torch.sigmoid(self.p5_ser_scale_logit)

    @staticmethod
    def _normalize_ser_delta(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Match per-image delta RMS to its reference so the bounded scale equals the real perturbation ratio."""
        reduce_dims = (0, 2, 3, 4)
        reference_rms = reference.detach().float().square().mean(reduce_dims, keepdim=True).sqrt()
        delta_rms = delta.detach().float().square().mean(reduce_dims, keepdim=True).sqrt()
        ratio = (reference_rms / (delta_rms + 1e-6)).to(dtype=delta.dtype)
        return delta * ratio

    def _build_guided_cls_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(x) != 4 or any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Safe Text-Route SER head expects P3/P4/P5 plus text embeddings")
        p4_delta = self.p4_ser_refine(self.p4_text_router(x[1], x[-1]))
        p5_delta = self.p5_ser_refine(self.p5_text_router(x[2], x[-1]))
        p4_delta = self._normalize_ser_delta(p4_delta, x[1])
        p5_delta = self._normalize_ser_delta(p5_delta, x[2])
        guided_p4 = x[1] + self.p4_ser_scale * p4_delta
        guided_p5 = x[2] + self.p5_ser_scale * p5_delta
        return [x[0], guided_p4, guided_p5]

    def _forward_head_with_guided_features(
        self,
        x,
        guided_cls_features,
        box_head,
        cls_head,
        mask_head=None,
        contrastive_head=None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[1]
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj

        # 文本路由只改变开放词汇分类特征；框回归、anchor特征和mask coefficient继续使用0725原始特征。
        boxes = torch.cat(
            [self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(cls_head[i](guided_cls_features[i]), cls_proj[i]),
                    x[-1],
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        preds = dict(boxes=boxes, scores=scores, feats=self._to_2d_feats(x[:3]))
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(x, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        guided_o2m = self._build_guided_cls_features(x)
        preds = self._forward_head_with_guided_features(x, guided_o2m, **self.one2many)

        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1].detach()]
            # 参考0803：P4/P5注意力只计算一次；one-to-one共享结果并detach，由one-to-many训练SER。
            guided_o2o = [feature.detach() for feature in guided_o2m]
            one2one = self._forward_head_with_guided_features(x_detach, guided_o2o, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        proto = self._forward_proto(x)
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = proto.detach()
        else:
            preds["proto"] = proto
        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        outputs = y if self.export else (y, preds)
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class SharedSafeTextRouteEnhancer(nn.Module):
    """在任务分支分流前，用安全文本路由统一增强P4/P5视觉特征。"""

    def __init__(
        self,
        p4_channels: int,
        p5_channels: int,
        text_dim: int = 512,
        num_heads: int = 8,
        n_win: int = 7,
        topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
    ) -> None:
        super().__init__()
        route_kwargs = dict(
            text_dim=text_dim,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            candidate_pool=candidate_pool,
            visual_keep=visual_keep,
        )
        self.p4_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=p4_channels, **route_kwargs)
        self.p5_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=p5_channels, **route_kwargs)
        # 参考0603：文本选路后的特征先经过SNN卷积重整，再以有界小残差写回Neck输出。
        self.p4_refine = MS_ConvBlock(p4_channels, mlp_ratio=1, sep_kernel_size=7)
        self.p5_refine = MS_ConvBlock(p5_channels, mlp_ratio=1, sep_kernel_size=7)

        self.residual_scale_max = 0.05
        initial_scale = 0.005
        initial_ratio = initial_scale / self.residual_scale_max
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.p4_scale_logit = nn.Parameter(torch.tensor(initial_logit))
        self.p5_scale_logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def p4_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.sigmoid(self.p4_scale_logit)

    @property
    def p5_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.sigmoid(self.p5_scale_logit)

    @staticmethod
    def _normalize_delta(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        reduce_dims = (0, 2, 3, 4)
        reference_rms = reference.detach().float().square().mean(reduce_dims, keepdim=True).sqrt()
        delta_rms = delta.detach().float().square().mean(reduce_dims, keepdim=True).sqrt()
        return delta * (reference_rms / (delta_rms + 1e-6)).to(dtype=delta.dtype)

    def forward(self, features: list[torch.Tensor], text_embeddings: torch.Tensor) -> list[torch.Tensor]:
        if len(features) != 3 or any(feature.ndim != 5 for feature in features):
            raise ValueError("SharedSafeTextRouteEnhancer expects P3/P4/P5 in [T,B,C,H,W]")
        p3, p4, p5 = features
        p4_delta = self.p4_refine(self.p4_text_router(p4, text_embeddings))
        p5_delta = self.p5_refine(self.p5_text_router(p5, text_embeddings))
        p4_delta = self._normalize_delta(p4_delta, p4)
        p5_delta = self._normalize_delta(p5_delta, p5)
        return [p3, p4 + self.p4_scale * p4_delta, p5 + self.p5_scale * p5_delta]


class YOLOESNNSegment22_2SharedTextRouteSER(YOLOESNNSegment22_2):
    """Segment22_2前置共享文本路由：增强后再进入分类、回归和mask coefficient分支。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        # 与上一版不同：直接从干净的Segment22_2构建三条任务路径，不继承任何分类专用SER逻辑。
        super().__init__(nc, nm, npr, embed, with_bn, reg_max, end2end, ch)
        self.shared_enhancer = SharedSafeTextRouteEnhancer(
            p4_channels=ch[1],
            p5_channels=ch[2],
            text_dim=embed,
            num_heads=ser_num_heads,
            n_win=ser_n_win,
            topk=ser_topk,
            candidate_pool=candidate_pool,
            visual_keep=visual_keep,
        )

    @property
    def p4_ser_scale(self) -> torch.Tensor:
        return self.shared_enhancer.p4_scale

    @property
    def p5_ser_scale(self) -> torch.Tensor:
        return self.shared_enhancer.p5_scale

    def _forward_head_with_shared_features(
        self,
        x,
        shared_features,
        box_head,
        cls_head,
        mask_head=None,
        contrastive_head=None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[1]
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj

        # 与0814分类专用版本不同：Head入口统一替换为[P3, guided_P4, guided_P5]，
        # 因此分类cv3、回归cv2和mask coefficient cv5看到同一套文本引导视觉特征。
        boxes = torch.cat(
            [
                self._aggregate_branch_output(box_head[i](shared_features[i])).view(bs, 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(cls_head[i](shared_features[i]), cls_proj[i]), x[-1]
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        preds = dict(boxes=boxes, scores=scores, feats=self._to_2d_feats(shared_features))
        if mask_head is not None:
            # 与0814不同：P4/P5的mask coefficient也共享增强特征；Proto仍在forward中读取原始P3。
            preds["mask_coefficient"] = self._forward_mask_coefficients(shared_features, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        if len(x) != 4 or any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Shared Safe Text-Route SER expects P3/P4/P5 plus text embeddings")

        # 与上一版不同：独立增强器位于Segment22_2三条路径之前，P4/P5注意力只计算一次。
        shared_o2m = self.shared_enhancer(x[:3], x[-1])
        preds = self._forward_head_with_shared_features(x, shared_o2m, **self.one2many)
        if self.end2end:
            x_detach = [feature.detach() for feature in x[:3]] + [x[-1].detach()]
            # 与0814相同：one-to-one复用同一份注意力结果并detach，不重复计算P4/P5 SER。
            shared_o2o = [feature.detach() for feature in shared_o2m]
            one2one = self._forward_head_with_shared_features(x_detach, shared_o2o, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        # 与共享P4/P5不同：Proto继续只使用原始P3，避免文本路由干扰高分辨率掩码边缘。
        proto = self._forward_proto(x)
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = proto.detach()
        else:
            preds["proto"] = proto
        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        outputs = y if self.export else (y, preds)
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class DirectSafeTextRouteEnhancer(nn.Module):
    """用Safe Text-SER正式变换P4/P5，不使用0819的外部残差、alpha或RMS归一化。"""

    def __init__(
        self,
        p4_channels: int,
        p5_channels: int,
        text_dim: int = 512,
        num_heads: int = 8,
        n_win: int = 7,
        topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
    ) -> None:
        super().__init__()
        route_kwargs = dict(
            text_dim=text_dim,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            candidate_pool=candidate_pool,
            visual_keep=visual_keep,
        )
        self.p4_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=p4_channels, **route_kwargs)
        self.p5_text_router = SafeTextGuidedBiLevelRoutingAttentionSNN(dim=p5_channels, **route_kwargs)
        # 与0819不同：沿用0603的“注意力后再卷积”，卷积输出直接成为新的Head特征。
        self.p4_refine = MS_ConvBlock(p4_channels, mlp_ratio=1, sep_kernel_size=7)
        self.p5_refine = MS_ConvBlock(p5_channels, mlp_ratio=1, sep_kernel_size=7)

    def forward(self, features: list[torch.Tensor], text_embeddings: torch.Tensor) -> list[torch.Tensor]:
        if len(features) != 3 or any(feature.ndim != 5 for feature in features):
            raise ValueError("DirectSafeTextRouteEnhancer expects P3/P4/P5 in [T,B,C,H,W]")
        p3, p4, p5 = features
        # 与0819不同：这里没有 P + alpha * RMS(delta)，P4/P5由Text-SER正式替换。
        p4_guided = self.p4_refine(self.p4_text_router(p4, text_embeddings))
        p5_guided = self.p5_refine(self.p5_text_router(p5, text_embeddings))
        return [p3, p4_guided, p5_guided]


class omnispike_SNNSegment22_2DirectTextRouteSER(YOLOESNNSegment22_2):
    """0603式双尺度Text-SER：P4/P5经Attention和MS_ConvBlock后直接进入共享Head。"""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        # 直接继承干净的Segment22_2，避免继承0819的残差写回实现。
        super().__init__(nc, nm, npr, embed, with_bn, reg_max, end2end, ch)
        self.direct_text_ser = DirectSafeTextRouteEnhancer(
            p4_channels=ch[1],
            p5_channels=ch[2],
            text_dim=embed,
            num_heads=ser_num_heads,
            n_win=ser_n_win,
            topk=ser_topk,
            candidate_pool=candidate_pool,
            visual_keep=visual_keep,
        )

    def _forward_head_with_direct_features(
        self,
        x,
        direct_features,
        box_head,
        cls_head,
        mask_head=None,
        contrastive_head=None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[1]
        boxes = torch.cat(
            [
                self._aggregate_branch_output(box_head[i](direct_features[i])).view(bs, 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                # COCO downstream transfer 时必须经过统一分类入口，才能使用固定类别 transfer_cls；
                # 开放词汇训练时该入口仍回退到原有的 cls_proj + 图文相似度计算。
                self._classification_logits(
                    cls_head[i](direct_features[i]), x[-1], cls_head, contrastive_head[i], i
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        preds = dict(boxes=boxes, scores=scores, feats=self._to_2d_feats(direct_features))
        if mask_head is not None:
            preds["mask_coefficient"] = self._forward_mask_coefficients(direct_features, mask_head)
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        if len(x) != 4 or any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Direct Safe Text-Route SER expects P3/P4/P5 plus text embeddings")

        # P4/P5注意力只执行一次，分类、回归和mask coefficient共享同一组直接替换特征。
        direct_o2m = self.direct_text_ser(x[:3], x[-1])
        preds = self._forward_head_with_direct_features(x, direct_o2m, **self.one2many)
        if self.end2end:
            x_detach = [feature.detach() for feature in x[:3]] + [x[-1].detach()]
            # one-to-one只复用并detach已有结果，不重复执行Text-SER。
            direct_o2o = [feature.detach() for feature in direct_o2m]
            one2one = self._forward_head_with_direct_features(x_detach, direct_o2o, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        # Proto始终读取原始P3，避免双尺度路由改变高分辨率mask基底。
        proto = self._forward_proto(x)
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = proto.detach()
        else:
            preds["proto"] = proto
        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        outputs = y if self.export else (y, preds)
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class TextGuidedChannelGateSNN(nn.Module):
    """Lightweight text-conditioned channel gate for SNN features.

    This module keeps P3 spatial/token structure unchanged and only rescales
    channels according to the text prompt. It is intentionally much cheaper
    than SER and is initialized close to an identity mapping.
    """

    def __init__(self, dim: int, text_dim: int = 512, hidden_dim: int | None = None, init_scale: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 16)
        self.text_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.gate_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Apply text-guided channel reweighting to [T, B, C, H, W] features."""
        if text_emb.ndim == 3:
            text_context = text_emb.mean(dim=1)
        elif text_emb.ndim == 2:
            text_context = text_emb
        else:
            raise ValueError(f"Expected text embeddings in [B,N,D] or [B,D] format, got {text_emb.shape}")

        gate = torch.sigmoid(self.text_gate(text_context)).view(1, x.shape[1], x.shape[2], 1, 1)
        return x * (1.0 + self.gate_scale * gate)


class YOLOESNNDetect2TextGuidedSER(YOLOESNNDetect2):
    """YOLOE-SNN head that applies text-guided SER only to classification features.

    P3/P4/P5 are passed unchanged to the box branch. Separate text-guided
    attention modules transform only the inputs to the visual embedding branch.
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 4,
        topk_text: int = 8,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        # Keep SER arguments before reg_max/end2end/ch because parse_model()
        # appends those final three values to the arguments supplied by YAML.
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        cls_attn = []
        for i, channels in enumerate(ch):
            if i == 0:
                # P3 is the highest-resolution feature map, so full regional routing
                # attention is disproportionately expensive here. Use a cheap text
                # channel gate and keep SER for the more semantic P4/P5 levels.
                cls_attn.append(TextGuidedChannelGateSNN(dim=channels, text_dim=embed))
            else:
                cls_attn.append(
                    TextGuidedBiLevelRoutingAttentionSNN(
                        dim=channels,
                        text_dim=embed,
                        num_heads=ser_num_heads,
                        n_win=ser_n_win,
                        topk=ser_topk,
                        topk_text=topk_text,
                    )
                )
        self.cls_attn = nn.ModuleList(cls_attn)

    def _forward_head_with_guided_features(
        self, x, guided_cls_features, box_head, cls_head, contrastive_head
    ):
        """Run one prediction branch using precomputed text-guided classification features."""
        if len(x) != 4:
            raise ValueError(f"Expected P3/P4/P5 plus text embeddings, got {len(x)} inputs")
        if box_head is None or cls_head is None:
            return dict()
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Text-guided YOLOE-SNN head expects P3/P4/P5 in [T,B,C,H,W] format")

        bs = x[0].shape[1]
        cls_proj = self.cls_proj if cls_head is self.cv3 else self.one2one_cls_proj

        # Localization is deliberately independent of all text-guided attention.
        boxes = torch.cat(
            [
                self._aggregate_branch_output(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )

        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                contrastive_head[i](
                    self._normalize_cls_feature(
                        self._aggregate_branch_output(cls_head[i](guided_cls_features[i])), cls_proj[i]
                    ),
                    x[-1],
                ).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.no = self.nc + self.reg_max * 4
        return dict(boxes=boxes, scores=scores, feats=self._to_2d_feats(x[:3]))

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Run a standalone branch, computing shared attention once for compatibility."""
        guided_cls_features = [self.cls_attn[i](x[i], x[-1]) for i in range(self.nl)]
        return self._forward_head_with_guided_features(
            x, guided_cls_features, box_head, cls_head, contrastive_head
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Compute shared SER once, then reuse its features in both end-to-end branches."""
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        if len(x) != 4:
            raise ValueError(f"Expected P3/P4/P5 plus text embeddings, got {len(x)} inputs")
        if any(feature.ndim != 5 for feature in x[:3]):
            raise ValueError("Text-guided YOLOE-SNN head expects P3/P4/P5 in [T,B,C,H,W] format")

        # SER is a shared perceptual stage. Dense one-to-many supervision trains
        # it, while the detached result preserves the original one-to-one stop-gradient semantics.
        guided_cls_features = [self.cls_attn[i](x[i], x[-1]) for i in range(self.nl)]
        preds = self._forward_head_with_guided_features(x, guided_cls_features, **self.one2many)

        if self.end2end:
            x_detach = [xi.detach() for xi in x[:3]] + [x[-1]]
            guided_detach = [feature.detach() for feature in guided_cls_features]
            one2one = self._forward_head_with_guided_features(x_detach, guided_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds

        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class YOLOESNNSegment2TextGuidedSER(YOLOESNNDetect2TextGuidedSER):
    """YOLOE-SNN open-vocabulary segmentation head with Text-Guided SER classification.

    分支设计：
      - 框回归分支：使用原始 SNN P3/P4/P5 特征；
      - mask 分支：原始 P3/P4/P5 先沿时间维求平均，再进入 ANN mask 读出层；
      - 类别嵌入分支：使用经过 Text-Guided SER 增强的 SNN P3/P4/P5 特征。

    重要的 SNN/ANN 分界：
      backbone、neck、框回归头和图像嵌入头处理五维 SNN 特征 [T, B, C, H, W]。
      当前 mask 分支并非全 SNN：`_to_2d_feats()` 会对 T 维求平均；在此之后，
      `cv5` 和 `proto` 使用普通二维 ANN 卷积，分别预测连续的 mask 系数和共享 mask 原型。
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = True,
        ser_num_heads: int = 8,
        ser_n_win: int = 7,
        ser_topk: int = 4,
        topk_text: int = 8,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        # Keep SER arguments before reg_max/end2end/ch because parse_model()
        # appends those final three values to the arguments supplied by YAML.
        super().__init__(
            nc=nc,
            embed=embed,
            with_bn=with_bn,
            ser_num_heads=ser_num_heads,
            ser_n_win=ser_n_win,
            ser_topk=ser_topk,
            topk_text=topk_text,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )
        self.nm = nm
        self.npr = npr

        # ======================== ANN MASK READOUT：原型分支 ========================
        # Proto 是 Ultralytics 标准 ANN 模块，由 Conv2d/BN/SiLU 和 ConvTranspose2d 组成。
        # 它不能直接接收 [T,B,C,H,W]。在 forward() 中，P3 先通过 `_to_2d_feats(...)[0]`
        # 转换成 BCHW，然后 Proto 生成 `nm` 张由所有实例共享的 mask 基底图。
        # 例如 imgsz=224、nm=32 时：P3 [B,C,28,28] -> proto [B,32,56,56]。
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)

        # ====================== ANN MASK READOUT：系数分支 ======================
        # cv5 同样是 ANN 而不是 SNN：这里的 Conv 是 Conv2d + BatchNorm2d + SiLU，
        # 最后的 nn.Conv2d 在 P3/P4/P5 的每个位置输出 `nm` 个连续系数。
        # 每个检测实例使用自己的系数向量，对共享的 `proto` 做线性组合：
        #     instance_mask_logits = mask_coefficient @ proto
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(channels, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for channels in ch
        )
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Return one-to-many branch components including mask coefficients."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Return one-to-one branch components including mask coefficients."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def _forward_head_with_guided_features(
        self, x, guided_cls_features, box_head, cls_head, mask_head=None, contrastive_head=None
    ):
        """Run detection prediction and append mask coefficients from raw features."""
        preds = super()._forward_head_with_guided_features(
            x, guided_cls_features, box_head, cls_head, contrastive_head
        )
        if mask_head is not None:
            bs = x[0].shape[1] if x[0].ndim == 5 else x[0].shape[0]

            # ============================ SNN -> ANN 分界 ============================
            # x[:3] 从 SNN neck 输出时为 [T,B,C,H,W]。对 T 求平均后得到普通的连续 BCHW
            # 特征图。从这一行开始，后续 mask_head/cv5 都属于 __init__() 中标记的
            # ANN mask 系数读出部分。
            feats_2d = self._to_2d_feats(x[:3])
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](feats_2d[i]).view(bs, self.nm, -1) for i in range(self.nl)],
                dim=2,
            )
        return preds

    def forward_head(self, x, box_head, cls_head, mask_head, contrastive_head):
        """Run a standalone branch, computing shared text-guided attention once for compatibility."""
        guided_cls_features = [self.cls_attn[i](x[i], x[-1]) for i in range(self.nl)]
        return self._forward_head_with_guided_features(
            x, guided_cls_features, box_head, cls_head, mask_head, contrastive_head
        )

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return open-vocabulary detection outputs, mask coefficients and mask prototypes."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs

        # ANN 原型读出：先对 SNN P3 的时间维求平均，再进入普通二维 Proto。
        # 生成的基底图由同一张图中的所有实例共享；cv5 为每个检测实例提供不同的组合系数。
        proto = self.proto(self._to_2d_feats(x[:3])[0])

        if isinstance(preds, dict):
            if self.end2end and not hasattr(self, "lrpc"):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode boxes/classes and concatenate mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process predictions with mask coefficients."""
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features and drop one-to-many mask branch for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None
            return


class YOLOESNNDetect3(YOLOEDetect):
    """SNN-to-ANN YOLOE head: aggregate temporal features first, then run standard YOLOE detection."""

    @staticmethod
    def _to_2d_feats(x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Convert SNN features from [T, B, C, H, W] to standard ANN [B, C, H, W] before any head branch runs."""
        return [xi.mean(0) if xi.ndim == 5 else xi for xi in x]

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        # 视觉提示编码 SAVPE 是 2D 模块，所以这里也统一先把时序特征压成 ANN 特征再复用 YOLOE 原逻辑。
        return super().get_vpe(self._to_2d_feats(x), vpe)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        # 正常检测路径：
        # 1. 进入 head 时，前三个输入仍然可能是 SNN 的 [T, B, C, H, W]；
        # 2. 在任何 cv2/cv3/cv4 分支运行之前，先整体聚合成普通 BCHW；
        # 3. 后续完全走标准 YOLOEDetect 的 ANN 检测逻辑。
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])

        feats_2d = self._to_2d_feats(x[:3])
        return super().forward(feats_2d + [x[-1]])

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        # prompt-free 路径同样先把时序特征转成 ANN，再执行标准 YOLOE 的 fuse 后推理分支。
        return super().forward_lrpc(self._to_2d_feats(x))


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    面向初学者的分割计算流程：
      1. 继承的 YOLOEDetect 分支预测检测框和开放词汇类别分数；
      2. `cv5` 为 P3/P4/P5 的每个候选位置预测 `nm` 个 mask 系数；
      3. `proto` 根据 P3 预测 `nm` 张由所有实例共享的全图 mask 基底；
      4. 最终实例 mask 是加权和 `coefficient @ proto`。

    原始 YOLOESegment 接收普通四维 BCHW 特征，因此 `cv5` 和 `proto` 都是 ANN mask
    readout，其中没有膜电位更新，也没有时间维。

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        # ======================== ANN MASK READOUT：原型分支 ========================
        # Proto 使用普通二维层，根据 P3 生成 `nm` 张共享 mask 基底图：
        # Conv2d -> ConvTranspose2d（上采样 2 倍）-> Conv2d -> Conv2d(nm)。
        # 每张基底图都是网络学习出的组合成分，不一定对应某种可解释的语义 mask。
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)

        # ====================== ANN MASK READOUT：系数分支 ======================
        # P3/P4/P5 每个尺度各有一条分支。最后的 1x1 Conv2d 为每个候选位置输出 `nm` 个
        # 连续组合权重。这些权重与 `proto` 组合，重建实例专属 mask，从而避免在每个位置
        # 直接预测一张完整 mask。
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # 继承的检测路径：检测框 + 文本对比分类分数 + mask 系数。
        # 当存在 mask_head/cv5 时，下面的 forward_head 会添加 mask 系数字段。
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size

            # 三个检测尺度的 ANN mask 系数读出。展平 H、W 后，每个检测候选对应一个
            # nm 维系数向量；沿 dim=2 拼接后得到 [B, nm, N_all_scales]。
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOEDetect functionality to include segmentation capabilities by integrating a prototype
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum regression value for bounding boxes. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None
