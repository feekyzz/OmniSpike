"""Text-guided Bi-Level Routing Attention for YOLOE-SNN classification features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_spikformer import BiLevelRoutingAttention_SNN, regional_routing_attention_torch


class TextGuidedBiLevelRoutingAttentionSNN(BiLevelRoutingAttention_SNN):
    """Build a temporally shared sparse region graph from visual and text similarities.

    Text changes both the discrete routing topology and a weak continuous residual.
    The residual is confidence-gated so low-similarity background regions are not
    forced to adopt a class prototype, and it supplies a gradient path that does
    not depend on the non-differentiable Top-K indices.
    """

    def __init__(
        self,
        dim: int,
        text_dim: int = 512,
        num_heads: int = 8,
        n_win: int = 7,
        topk: int = 4,
        topk_text: int = 8,
        qk_scale: float | None = None,
        side_dwconv: int = 3,
        auto_pad: bool = True,
        route_mode: str = "time_avg",
        text_temperature: float = 0.1,
        gate_threshold: float = 0.0,
        initial_text_logit: float = -4.0,
        initial_residual_scale: float = 0.1,
        fixed_text_strength: float | None = None,
        use_text_residual: bool = True,
        use_text_attention_gate: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            qk_scale=qk_scale,
            side_dwconv=side_dwconv,
            auto_pad=auto_pad,
            route_mode=route_mode,
        )
        if topk_text < 1:
            raise ValueError("topk_text must be positive")
        if text_temperature <= 0:
            raise ValueError("text_temperature must be positive")

        self.text_dim = text_dim
        self.topk_text = topk_text
        self.text_temperature = text_temperature
        self.gate_threshold = gate_threshold
        self.text_proj = nn.Linear(text_dim, dim, bias=False)
        self.text_map_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.text_strength_logit = nn.Parameter(torch.tensor(float(initial_text_logit)))
        self.text_residual_scale = nn.Parameter(torch.tensor(float(initial_residual_scale)))
        if fixed_text_strength is not None and not 0.0 <= fixed_text_strength <= 1.0:
            raise ValueError("fixed_text_strength must be in [0, 1]")
        self.fixed_text_strength = fixed_text_strength
        self.use_text_residual = use_text_residual
        self.use_text_attention_gate = use_text_attention_gate

    def _prepare_text(
        self,
        text_emb: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)
        if text_emb.ndim != 3:
            raise ValueError(f"text_emb must be [N,D] or [B,N,D], got {tuple(text_emb.shape)}")
        if text_emb.shape[-1] != self.text_dim:
            raise ValueError(f"Expected text dim {self.text_dim}, got {text_emb.shape[-1]}")
        if text_emb.shape[0] == 1 and batch_size > 1:
            text_emb = text_emb.expand(batch_size, -1, -1)
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"Text batch {text_emb.shape[0]} does not match visual batch {batch_size}")
        return text_emb.to(device=device, dtype=dtype)

    def _build_text_guided_region_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = q.shape
        region_h = max(height // self.n_win, 1)
        region_w = max(width // self.n_win, 1)
        region_size = (region_h, region_w)

        q_pool = F.avg_pool2d(q, region_size, ceil_mode=True, count_include_pad=False)
        k_pool = F.avg_pool2d(k, region_size, ceil_mode=True, count_include_pad=False)
        grid_h, grid_w = q_pool.shape[-2:]
        q_region = q_pool.flatten(2).transpose(1, 2)
        k_region = k_pool.flatten(2).transpose(1, 2)
        q_norm = F.normalize(q_region, dim=-1, eps=1e-6)
        k_norm = F.normalize(k_region, dim=-1, eps=1e-6)

        # Both components are cosine graphs and therefore share a controlled scale.
        visual_graph = q_norm @ k_norm.transpose(-1, -2)

        text_emb = self._prepare_text(text_emb, batch_size, q.device, q.dtype)
        text_proto = F.normalize(self.text_proj(text_emb), dim=-1, eps=1e-6)
        region_text_similarity = q_norm @ text_proto.transpose(-1, -2)
        selected_count = min(self.topk_text, text_proto.shape[1])
        top_values, top_indices = torch.topk(region_text_similarity, selected_count, dim=-1)

        batch_index = torch.arange(batch_size, device=q.device)[:, None, None]
        selected_text = text_proto[batch_index, top_indices]
        weights = F.softmax(top_values / self.text_temperature, dim=-1)
        text_context = (weights.unsqueeze(-1) * selected_text).sum(dim=-2)
        text_context = F.normalize(text_context, dim=-1, eps=1e-6)

        # Suppress semantic routing for regions whose best text match is weak.
        confidence = torch.sigmoid((top_values[..., 0] - self.gate_threshold) / self.text_temperature)
        semantic_graph = text_context @ text_context.transpose(-1, -2)
        semantic_graph = semantic_graph * confidence.unsqueeze(-1) * confidence.unsqueeze(-2)

        text_strength = (
            torch.sigmoid(self.text_strength_logit)
            if self.fixed_text_strength is None
            else q.new_tensor(self.fixed_text_strength)
        )
        fused_graph = visual_graph + text_strength * semantic_graph
        route_count = min(self.topk, fused_graph.shape[-1])
        route_indices = torch.topk(fused_graph, route_count, dim=-1).indices
        route_indices = route_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        text_context = text_context * confidence.unsqueeze(-1)
        text_map = text_context.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)
        text_map = F.interpolate(text_map, size=(height, width), mode="nearest")
        # 连续置信度门控为 text_proj 提供可微梯度；它只缩放聚合结果，不把文本向量写入视觉特征。
        text_attention_gate = confidence.reshape(batch_size, 1, grid_h, grid_w)
        text_attention_gate = F.interpolate(text_attention_gate, size=(height, width), mode="nearest")
        return route_indices, region_size, text_map, text_attention_gate, text_strength

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        ret_attn_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected SNN feature [T,B,C,H,W], got {tuple(x.shape)}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.route_mode == "time_avg":
            q_route, k_route = q.mean(0), k.mean(0)
        elif self.route_mode == "last":
            q_route, k_route = q[-1], k[-1]
        else:
            raise ValueError(f"Unsupported route_mode: {self.route_mode}")

        route_indices, region_size, text_map, text_attention_gate, text_strength = self._build_text_guided_region_graph(
            q_route, k_route, text_emb
        )
        text_residual = self.text_map_proj(text_map) if self.use_text_residual else None
        lepe = self.lepe(v)
        outputs = []
        attention_masks = []
        for time_index in range(x.shape[0]):
            output, attention = regional_routing_attention_torch(
                query=q[time_index],
                key=k[time_index],
                value=v[time_index],
                scale=self.scale,
                region_graph=route_indices,
                region_size=region_size,
                auto_pad=self.auto_pad,
            )
            output = output + lepe[time_index]
            if self.use_text_attention_gate:
                output = output * (1.0 + text_strength * text_attention_gate)
            if self.use_text_residual:
                output = output + text_strength * self.text_residual_scale * text_residual
            outputs.append(output)
            if ret_attn_mask:
                attention_masks.append(attention)

        result = self.out_proj(torch.stack(outputs, dim=0))
        return (result, attention_masks) if ret_attn_mask else result


class SafeTextGuidedBiLevelRoutingAttentionSNN(BiLevelRoutingAttention_SNN):
    """Select safe visual routes with calibrated text guidance while keeping visual values spike-driven.

    Text is only allowed to rerank a visually plausible candidate pool. A fixed number of the strongest
    visual routes is always retained, so a vocabulary change cannot replace the complete visual topology.
    """

    def __init__(
        self,
        dim: int,
        text_dim: int = 512,
        num_heads: int = 8,
        n_win: int = 7,
        topk: int = 4,
        candidate_pool: int = 8,
        visual_keep: int = 2,
        qk_scale: float | None = None,
        side_dwconv: int = 3,
        auto_pad: bool = True,
        route_mode: str = "time_avg",
        margin_threshold: float = 0.02,
        margin_temperature: float = 0.05,
        initial_route_strength: float = 0.05,
        max_route_strength: float = 0.25,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            qk_scale=qk_scale,
            side_dwconv=side_dwconv,
            auto_pad=auto_pad,
            route_mode=route_mode,
        )
        if candidate_pool < topk:
            raise ValueError("candidate_pool must be greater than or equal to topk")
        if not 0 <= visual_keep <= topk:
            raise ValueError("visual_keep must be in [0, topk]")
        if margin_temperature <= 0:
            raise ValueError("margin_temperature must be positive")
        if not 0 < initial_route_strength < max_route_strength:
            raise ValueError("initial_route_strength must be in (0, max_route_strength)")

        self.text_dim = text_dim
        self.candidate_pool = candidate_pool
        self.visual_keep = visual_keep
        self.margin_threshold = margin_threshold
        self.margin_temperature = margin_temperature
        self.max_route_strength = max_route_strength
        self.text_proj = nn.Linear(text_dim, dim, bias=False)
        initial_ratio = initial_route_strength / max_route_strength
        self.route_strength_logit = nn.Parameter(torch.tensor(math.log(initial_ratio / (1.0 - initial_ratio))))

    @property
    def route_strength(self) -> torch.Tensor:
        """Return the bounded strength used only for text-based candidate reranking."""
        return self.max_route_strength * torch.sigmoid(self.route_strength_logit)

    def _prepare_text(
        self,
        text_emb: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)
        if text_emb.ndim != 3:
            raise ValueError(f"text_emb must be [N,D] or [B,N,D], got {tuple(text_emb.shape)}")
        if text_emb.shape[-1] != self.text_dim:
            raise ValueError(f"Expected text dim {self.text_dim}, got {text_emb.shape[-1]}")
        if text_emb.shape[0] == 1 and batch_size > 1:
            text_emb = text_emb.expand(batch_size, -1, -1)
        if text_emb.shape[0] != batch_size:
            raise ValueError(f"Text batch {text_emb.shape[0]} does not match visual batch {batch_size}")
        return text_emb.to(device=device, dtype=dtype)

    def _build_safe_text_routes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor]:
        batch_size, _, height, width = q.shape
        region_h = max(height // self.n_win, 1)
        region_w = max(width // self.n_win, 1)
        region_size = (region_h, region_w)

        q_pool = F.avg_pool2d(q, region_size, ceil_mode=True, count_include_pad=False)
        k_pool = F.avg_pool2d(k, region_size, ceil_mode=True, count_include_pad=False)
        q_region = F.normalize(q_pool.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        k_region = F.normalize(k_pool.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        visual_graph = q_region @ k_region.transpose(-1, -2)

        text_emb = self._prepare_text(text_emb, batch_size, q.device, q.dtype)
        text_proto = F.normalize(self.text_proj(text_emb), dim=-1, eps=1e-6)
        source_text_similarity = q_region @ text_proto.transpose(-1, -2)

        # 与旧Text-SER不同：置信度使用top1-top2间隔，降低验证词表数量变化造成的最大值偏移。
        if text_proto.shape[1] >= 2:
            top_values, top_indices = torch.topk(source_text_similarity, 2, dim=-1)
            confidence = torch.sigmoid(
                (top_values[..., 0] - top_values[..., 1] - self.margin_threshold) / self.margin_temperature
            )
        else:
            top_values, top_indices = torch.topk(source_text_similarity, 1, dim=-1)
            confidence = torch.sigmoid((top_values[..., 0] - self.margin_threshold) / self.margin_temperature)

        batch_index = torch.arange(batch_size, device=q.device)[:, None]
        source_text = text_proto[batch_index, top_indices[..., 0]]
        # 文本明确回答“目标区域是否与源区域选择的文本语义一致”，而不是比较两个文本上下文是否相似。
        text_route_graph = source_text @ k_region.transpose(-1, -2)

        region_count = visual_graph.shape[-1]
        route_count = min(self.topk, region_count)
        candidate_count = min(max(self.candidate_pool, route_count), region_count)
        visual_keep = min(self.visual_keep, route_count)
        text_keep = route_count - visual_keep

        candidate_visual_scores, candidate_indices = torch.topk(visual_graph, candidate_count, dim=-1)
        candidate_text_scores = torch.gather(text_route_graph, dim=-1, index=candidate_indices)
        rerank_scores = candidate_visual_scores + (
            self.route_strength * confidence.unsqueeze(-1) * candidate_text_scores
        )

        # 至少保留visual_keep条纯视觉路由，文本只能在剩余视觉候选池内选择，不能推翻全部视觉拓扑。
        selected_routes = [candidate_indices[..., :visual_keep]] if visual_keep else []
        if text_keep:
            rerank_scores = rerank_scores.clone()
            if visual_keep:
                rerank_scores[..., :visual_keep] = torch.finfo(rerank_scores.dtype).min
            text_positions = torch.topk(rerank_scores, text_keep, dim=-1).indices
            selected_routes.append(torch.gather(candidate_indices, dim=-1, index=text_positions))
        route_indices = torch.cat(selected_routes, dim=-1)
        route_indices = route_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # 连续置信度门控不写入任何文本向量，只为文本投影和路由强度提供可微训练路径。
        grid_h, grid_w = q_pool.shape[-2:]
        text_gate = confidence.reshape(batch_size, 1, grid_h, grid_w)
        text_gate = F.interpolate(text_gate, size=(height, width), mode="nearest")
        return route_indices, region_size, text_gate

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        ret_attn_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected SNN feature [T,B,C,H,W], got {tuple(x.shape)}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.route_mode == "time_avg":
            q_route, k_route = q.mean(0), k.mean(0)
        elif self.route_mode == "last":
            q_route, k_route = q[-1], k[-1]
        else:
            raise ValueError(f"Unsupported route_mode: {self.route_mode}")

        route_indices, region_size, text_gate = self._build_safe_text_routes(q_route, k_route, text_emb)
        lepe = self.lepe(v)
        outputs = []
        attention_masks = []
        for time_index in range(x.shape[0]):
            output, attention = regional_routing_attention_torch(
                query=q[time_index],
                key=k[time_index],
                value=v[time_index],
                scale=self.scale,
                region_graph=route_indices,
                region_size=region_size,
                auto_pad=self.auto_pad,
            )
            output = output + lepe[time_index]
            output = output * (1.0 + self.route_strength * text_gate)
            outputs.append(output)
            if ret_attn_mask:
                attention_masks.append(attention)

        result = self.out_proj(torch.stack(outputs, dim=0))
        return (result, attention_masks) if ret_attn_mask else result


__all__ = ("SafeTextGuidedBiLevelRoutingAttentionSNN", "TextGuidedBiLevelRoutingAttentionSNN")
