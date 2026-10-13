"""Public-prefix evidence for the original Qwen language generation head.

No target argument exists in GenerativeEvidence.forward. All entity soft slots
are computed before matching; annotations are accepted only by auxiliary_loss.
The intermediate box heads supervise latent states, never constrain the final
language generator's entity/region output or replace its CE objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import minimum_cost_assignment as _hungarian

ENTITY_TYPES = ("PER", "LOC", "ORG", "MIS")
DEFAULT_CONFIG = dict(
    hidden_dim=256,
    entity_queries=16,
    attention_heads=8,
    reader_layers=2,
    feedforward_dim=1024,
    discovery_queries=128,
    completion_queries=128,
    max_images=16,
    dropout=0.1,
    support_dropout=0.1,
    mc_samples=8,
    uncertainty_eta=0.5,
    min_receiver_gate=0.1,
    memory_epsilon=1e-6,
    mode="full",
    cross_image_messages=True,
    uncertainty_guidance=True,
    completion=True,
    conditioned_completion=True,
    ner_negative_weight=0.2,
    cost_entity_type=1.0,
    cost_entity_span=1.0,
    weight_entity_type=1.0,
    weight_entity_span=1.0,
    cost_support=1.0,
    cost_l1=5.0,
    cost_giou=2.0,
    weight_support=1.0,
    weight_l1=5.0,
    weight_giou=2.0,
    no_support_weight=0.1,
    support_loss_normalization="positive_negative_balanced",
    loss_discovery=1.0,
    loss_refined=1.0,
    loss_final=1.0,
    loss_completion=1.0,
    residual_threshold=0.5,
    iou_threshold=0.5,
)


def binary_entropy(p: Tensor) -> Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def support_uncertainty(samples: Tensor, eta: float) -> tuple[Tensor, Tensor]:
    samples = samples.detach().float()
    mean = samples.mean(0)
    ambiguity = binary_entropy(mean) / math.log(2)
    disagreement = ((binary_entropy(mean) - binary_entropy(samples).mean(0)) / math.log(2)).clamp(
        0, 1
    )
    return mean, (eta * ambiguity + (1 - eta) * disagreement).clamp(0, 1)


class ReaderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            c.hidden_dim, c.attention_heads, c.dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            c.hidden_dim, c.attention_heads, c.dropout, batch_first=True
        )
        self.norms = nn.ModuleList(nn.LayerNorm(c.hidden_dim) for _ in range(3))
        self.ffn = nn.Sequential(
            nn.Linear(c.hidden_dim, c.feedforward_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.feedforward_dim, c.hidden_dim),
        )
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, q, visual, padding_mask):
        z = self.norms[0](q)
        q = q + self.dropout(self.self_attention(z, z, z, need_weights=False)[0])
        z = self.norms[1](q)
        q = q + self.dropout(
            self.cross_attention(
                z, visual, visual, key_padding_mask=padding_mask, need_weights=False
            )[0]
        )
        return q + self.dropout(self.ffn(self.norms[2](q)))


class SharedReader(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList(ReaderLayer(c) for _ in range(c.reader_layers))
        self.norm = nn.LayerNorm(c.hidden_dim)

    def forward(self, queries, visual, padding_mask):
        e, m, k, d = queries.shape
        if not m:
            return queries
        q = queries.reshape(e * m, k, d)
        v = visual.unsqueeze(0).expand(e, -1, -1, -1).reshape(e * m, visual.shape[1], d)
        mask = padding_mask.unsqueeze(0).expand(e, -1, -1).reshape(e * m, -1)
        for layer in self.layers:
            q = layer(q, v, mask)
        return self.norm(q).reshape(e, m, k, d)


class SupportHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.first = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.last = nn.Linear(c.hidden_dim, 1)
        self.dropout = c.support_dropout

    def forward(self, z):
        return self.last(
            F.dropout(F.gelu(self.first(z)), p=self.dropout, training=self.training)
        ).squeeze(-1)

    def mc_probabilities(self, z, samples):
        # Only this cheap head is resampled; reader and Qwen never repeat.
        x = F.gelu(self.first(z)).unsqueeze(0).expand(samples, *z.shape)
        return self.last(F.dropout(x, p=self.dropout, training=True)).squeeze(-1).sigmoid()


class CrossImageRefinement(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.k = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.entity = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.value = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)
        self.context = nn.Linear(c.hidden_dim, c.hidden_dim, bias=False)
        self.null_score = nn.Parameter(torch.zeros(()))

    def forward(self, z, entity, mean, uncertainty):
        e, m, k, d = z.shape
        flat = z.flatten(1, 2)
        reliability = (
            mean.detach() * (1 - uncertainty.detach())
            if self.c.uncertainty_guidance
            else mean.detach()
        )
        if not m:
            return z, dict(
                donor_reliability=reliability,
                receiver_gate=uncertainty,
                cross_attention=z.new_zeros(e, 0, 1),
                cross_message=z,
            )
        condition = self.entity(entity).unsqueeze(1)
        scores = torch.matmul(
            self.q(flat + condition).float(), self.k(flat + condition).float().transpose(-1, -2)
        ) / math.sqrt(d)
        donor = reliability.flatten(1, 2)
        scores = (
            scores
            + donor.clamp_min(torch.finfo(scores.dtype).tiny)
            .log()
            .masked_fill(donor <= 0, -torch.inf)[:, None, :]
        )
        image_indices = torch.arange(m, device=z.device).repeat_interleave(k)
        same_image = image_indices[:, None] == image_indices[None, :]
        scores = scores.masked_fill(same_image, -torch.inf)
        attention = torch.cat((scores, self.null_score.expand(e, m * k, 1)), dim=-1).softmax(-1)
        message = torch.matmul(attention[..., :-1].to(dtype=z.dtype), self.value(flat)).reshape_as(
            z
        )
        if not self.c.cross_image_messages:
            message = message * 0
        gate = self.c.min_receiver_gate + (1 - self.c.min_receiver_gate) * uncertainty.detach()
        if not self.c.uncertainty_guidance:
            gate = torch.ones_like(gate)
        return z + gate[..., None].to(dtype=z.dtype) * self.context(message), dict(
            donor_reliability=reliability,
            receiver_gate=gate,
            cross_attention=attention,
            cross_message=message,
        )


class EvidenceMemory(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.box_embedding = nn.Sequential(
            nn.Linear(4, c.hidden_dim), nn.GELU(), nn.Linear(c.hidden_dim, c.hidden_dim)
        )
        self.image_embedding = nn.Embedding(c.max_images, c.hidden_dim)
        self.null_token = nn.Parameter(torch.randn(1, 1, c.hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            c.hidden_dim, c.attention_heads, dropout=c.dropout, batch_first=True
        )

    def tokens(self, stage, entity_states=None):
        z, boxes, p = stage["states"], stage["boxes"], stage["logits"].sigmoid()
        e, m, k, d = z.shape
        image = self.image_embedding(torch.arange(m, device=z.device))[None, :, None, :]
        memory = (z + self.box_embedding(boxes.to(dtype=z.dtype)) + image).flatten(1, 2)
        if entity_states is not None:
            memory = memory + entity_states[:, None, :]
        null = self.null_token.expand(e, -1, -1)
        if entity_states is not None:
            null = null + entity_states[:, None, :]
        memory = torch.cat((memory, null), dim=1)
        bias = torch.cat(
            ((self.c.memory_epsilon + p.detach().flatten(1, 2)).log(), p.new_zeros(e, 1)), dim=1
        )
        return memory, bias

    def forward(self, queries, refined, return_weights=False):
        memory, bias = self.tokens(refined)
        e, m, k, d = queries.shape
        if not m:
            return queries, None
        q = queries.flatten(1, 2)
        mask = (
            bias[:, None, :].expand(e, q.shape[1], -1).repeat_interleave(self.c.attention_heads, 0)
        )
        context, weights = self.attention(
            q, memory, memory, attn_mask=mask, need_weights=return_weights
        )
        return queries + context.reshape_as(queries), weights


class GenerativeEvidence(nn.Module):
    """Single-post differentiable memory built exclusively from public states.

    Entity soft queries condition discovery, refinement, and completion readers.
    memory has E*(M*Kfinal+1) rows, including one explicit null per entity slot.
    memory_bias supplies detached support log-priors for the downstream reader.
    Neither returned auxiliary boxes nor finite query budgets truncate the LM.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, backbone_dim: int = 3584):
        super().__init__()
        values = {**DEFAULT_CONFIG, **dict(config or {})}
        self.config = values
        self.c = SimpleNamespace(**values)
        c = self.c
        if c.mode != "full":
            raise ValueError("The evidence module uses mode=full")
        for key in (
            "hidden_dim",
            "entity_queries",
            "attention_heads",
            "reader_layers",
            "feedforward_dim",
            "discovery_queries",
            "completion_queries",
            "max_images",
            "mc_samples",
        ):
            if type(values[key]) is not int or values[key] < 1:
                raise ValueError(key + " must be positive integer")
        if c.hidden_dim % c.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if (
            not 0 <= c.uncertainty_eta <= 1
            or not 0 < c.min_receiver_gate < 1
            or c.memory_epsilon <= 0
        ):
            raise ValueError("invalid routing settings")
        if not 0 <= c.dropout < 1 or not 0 <= c.support_dropout < 1:
            raise ValueError("invalid dropout")
        if c.support_loss_normalization not in (
            "positive_negative_balanced",
            "legacy_weighted_mean",
        ):
            raise ValueError("invalid support normalization")
        for key in ("ner_negative_weight", "no_support_weight"):
            if not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(key + " must be positive and finite")
        for key in (
            "cost_entity_type",
            "cost_entity_span",
            "weight_entity_type",
            "weight_entity_span",
            "cost_support",
            "cost_l1",
            "cost_giou",
            "weight_support",
            "weight_l1",
            "weight_giou",
            "loss_discovery",
            "loss_refined",
            "loss_final",
            "loss_completion",
        ):
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(key + " must be nonnegative and finite")
        if not 0 <= c.residual_threshold <= 1 or not 0 <= c.iou_threshold <= 1:
            raise ValueError("invalid residual threshold")
        if type(backbone_dim) is not int or backbone_dim < 1:
            raise ValueError("backbone_dim must be positive integer")
        self.backbone_dim = int(backbone_dim)
        d = c.hidden_dim
        self.text_projection = nn.Sequential(nn.Linear(backbone_dim, d), nn.LayerNorm(d))
        self.visual_projection = nn.Sequential(nn.Linear(backbone_dim, d), nn.LayerNorm(d))
        self.spatial_embedding = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.image_embedding = nn.Embedding(c.max_images, d)
        self.entity_queries = nn.Parameter(torch.randn(c.entity_queries, d) * 0.02)
        self.entity_attention = nn.MultiheadAttention(
            d, c.attention_heads, c.dropout, batch_first=True
        )
        self.entity_norm = nn.LayerNorm(d)
        self.entity_ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.entity_final_norm = nn.LayerNorm(d)
        self.start_query = nn.Linear(d, d)
        self.end_query = nn.Linear(d, d)
        self.word_key = nn.Linear(d, d)
        self.type_head = nn.Linear(d, 5)
        self.entity_projection = nn.Linear(d, d)
        self.discovery_queries = nn.Parameter(torch.randn(c.discovery_queries, d) * 0.02)
        self.completion_queries = nn.Parameter(torch.randn(c.completion_queries, d) * 0.02)
        self.reader = SharedReader(c)
        self.support_head = SupportHead(c)
        self.box_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.refinement = CrossImageRefinement(c)
        self.evidence_memory = EvidenceMemory(c)
        self.final_attention = nn.MultiheadAttention(
            d, c.attention_heads, c.dropout, batch_first=True
        )
        self.final_norm = nn.LayerNorm(d)

    def _decode(self, states):
        # Geometry remains FP32 even when the trainable evidence reader is BF16.
        raw = self.box_head(states).float().sigmoid()
        lower = raw[..., :2]
        upper = lower + (1 - lower) * raw[..., 2:]
        return dict(
            states=states, logits=self.support_head(states), boxes=torch.cat((lower, upper), dim=-1)
        )

    def _visual(self, visual_states, visual_positions, device, dtype):
        if len(visual_states) != len(visual_positions) or len(visual_states) > self.c.max_images:
            raise ValueError("image count/positions mismatch")
        m = len(visual_states)
        maximum = max((len(v) for v in visual_states), default=0)
        visual = torch.zeros(m, maximum, self.c.hidden_dim, device=device, dtype=dtype)
        mask = torch.ones(m, maximum, device=device, dtype=torch.bool)
        for i, (states, positions) in enumerate(zip(visual_states, visual_positions)):
            if (
                states.ndim != 2
                or states.shape[1] != self.backbone_dim
                or not len(states)
                or positions.shape != (len(states), 2)
            ):
                raise ValueError("invalid visual states/grid positions")
            if states.device != device or positions.device != device:
                raise ValueError("public state tensors must share module device")
            if (
                not torch.isfinite(states).all()
                or not torch.isfinite(positions).all()
                or bool(((positions < 0) | (positions > 1)).any())
            ):
                raise ValueError("nonfinite visual states or invalid normalized xy")
            value = (
                self.visual_projection(states.to(dtype=dtype))
                + self.spatial_embedding(positions.to(dtype=dtype))
                + self.image_embedding.weight[i]
            )
            visual[i, : len(value)] = value
            mask[i, : len(value)] = False
        return visual, mask

    def forward(
        self,
        public_text_states: Tensor,
        visual_states: Sequence[Tensor],
        visual_positions: Sequence[Tensor],
        image_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        # There is deliberately no annotations/gold/labels/answer argument.
        if (
            not isinstance(public_text_states, Tensor)
            or public_text_states.ndim != 2
            or public_text_states.shape[1] != self.backbone_dim
            or not len(public_text_states)
        ):
            raise ValueError("public_text_states must be nonempty [words,backbone_dim]")
        device = self.entity_queries.device
        dtype = self.entity_queries.dtype
        if public_text_states.device != device or not torch.isfinite(public_text_states).all():
            raise ValueError("public text state device/nonfinite mismatch")
        m = len(visual_states)
        ids = list(image_ids) if image_ids is not None else [str(i) for i in range(m)]
        if len(ids) != m or len(set(ids)) != m or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("image IDs must be unique ordered nonempty strings")
        text = self.text_projection(public_text_states.to(dtype=dtype))
        queries = self.entity_queries.unsqueeze(0)
        states, word_attention = self.entity_attention(
            queries, text.unsqueeze(0), text.unsqueeze(0), need_weights=True
        )
        states = self.entity_norm(queries + states)[0]
        states = self.entity_final_norm(states + self.entity_ffn(states))
        keys = self.word_key(text)
        entity = dict(
            states=states,
            start_logits=self.start_query(states) @ keys.T / math.sqrt(self.c.hidden_dim),
            end_logits=self.end_query(states) @ keys.T / math.sqrt(self.c.hidden_dim),
            type_logits=self.type_head(states),
            word_attention=word_attention[0],
        )
        visual, mask = self._visual(visual_states, visual_positions, device, dtype)
        e = len(states)
        condition = self.entity_projection(states)[:, None, None, :]
        discovery = self.discovery_queries[None, None, :, :].expand(e, m, -1, -1) + condition
        stage0 = self._decode(self.reader(discovery, visual, mask))
        stages = {"discovery": stage0}
        diagnostics = dict(
            mode=self.c.mode,
            visual_mask=mask,
            public_words=len(text),
            images=m,
            entity_slots=e,
            mc_reader_repeated=False,
            annotations_as_forward_input=False,
        )
        with torch.no_grad():
            samples = self.support_head.mc_probabilities(
                stage0["states"].detach(), self.c.mc_samples
            )
            mean, uncertainty = support_uncertainty(samples, self.c.uncertainty_eta)
        updated, route = self.refinement(stage0["states"], states, mean, uncertainty)
        stage1 = self._decode(self.reader(updated, visual, mask))
        stages["refined"] = stage1
        diagnostics.update(route, support_mean=mean, support_uncertainty=uncertainty)
        final_states = stage1["states"]
        if self.c.completion:
            completion = self.completion_queries[None, None, :, :].expand(e, m, -1, -1) + condition
            if self.c.conditioned_completion:
                completion, _ = self.evidence_memory(completion, stage1)
            stagec = self._decode(self.reader(completion, visual, mask))
            stages["completion"] = stagec
            final_states = torch.cat((final_states, stagec["states"]), dim=2)
        if m:
            flat = final_states.flatten(0, 1)
            final_states = self.final_norm(
                flat + self.final_attention(flat, flat, flat, need_weights=False)[0]
            ).reshape_as(final_states)
        stages["final"] = self._decode(final_states)
        memory, bias = self.evidence_memory.tokens(stages["final"], states)
        diagnostics["memory_slots"] = memory.shape[1] * e
        return dict(
            memory=memory.flatten(0, 1),
            memory_bias=bias.flatten(0, 1),
            entity=entity,
            stages=stages,
            diagnostics=diagnostics,
            num_words=len(text),
            num_images=m,
            image_ids=ids,
            config=dict(self.config),
        )

    def auxiliary_loss(self, output, gold_entities, num_words, num_images):
        return auxiliary_loss(output, gold_entities, num_words, num_images)


# The detached matching helpers below are independent of the public forward.


def _pairwise_iou(a: Tensor, b: Tensor, giou=False) -> Tensor:
    a = a.float()
    b = b.float()
    lower = torch.maximum(a[..., :, None, :2], b[..., None, :, :2])
    upper = torch.minimum(a[..., :, None, 2:], b[..., None, :, 2:])
    side = (upper - lower).clamp(min=0)
    intersection = side[..., 0] * side[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a[..., :, None] + area_b[..., None, :] - intersection
    tiny = torch.finfo(torch.float32).tiny
    iou = intersection / union.clamp_min(tiny)
    if not giou:
        return iou
    enclosure_side = (
        torch.maximum(a[..., :, None, 2:], b[..., None, :, 2:])
        - torch.minimum(a[..., :, None, :2], b[..., None, :, :2])
    ).clamp(min=0)
    enclosure = enclosure_side[..., 0] * enclosure_side[..., 1]
    return iou - (enclosure - union) / enclosure.clamp_min(tiny)


def _validate_gold(output, gold_entities, num_words, num_images):
    if num_words != output["num_words"] or num_images != output["num_images"]:
        raise ValueError("aux word/image dimensions differ from public forward")
    if not isinstance(gold_entities, (list, tuple)):
        raise TypeError("gold_entities must be an explicit list")
    if len(gold_entities) > len(output["entity"]["states"]):
        raise ValueError(
            "entity annotations exceed soft-slot capacity; increase entity_queries without truncating gold"
        )
    result = []
    seen = set()
    for entity in gold_entities:
        if not isinstance(entity, Mapping):
            raise ValueError("entity must be mapping")
        span = entity.get("token_span", [entity.get("start"), entity.get("end")])
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1] <= num_words
        ):
            raise ValueError("invalid original-word half-open gold span")
        if entity.get("type") not in ENTITY_TYPES:
            raise ValueError("unknown canonical entity type")
        if (span[0], span[1], entity["type"]) in seen:
            raise ValueError("duplicate gold entity")
        seen.add((span[0], span[1], entity["type"]))
        if not isinstance(entity.get("regions"), list):
            raise ValueError("regions must be explicit list; unknown is not no-visual")
        boxes = [[] for _ in range(num_images)]
        for region in entity["regions"]:
            if not isinstance(region, Mapping):
                raise ValueError("region must be mapping")
            image_index = region.get("image_index")
            if image_index is None:
                image_id = region.get("image_id")
                if image_id not in output["image_ids"]:
                    raise ValueError("unknown gold image ID")
                image_index = output["image_ids"].index(image_id)
            if type(image_index) is not int or not 0 <= image_index < num_images:
                raise ValueError("invalid gold image index")
            coords = region.get("box", region.get("bbox"))
            if (
                not isinstance(coords, (list, tuple))
                or len(coords) != 4
                or any(
                    isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                    for x in coords
                )
            ):
                raise ValueError("box requires finite normalized xyxy")
            x0, y0, x1, y1 = coords
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("invalid normalized gold box")
            boxes[image_index].append(list(map(float, coords)))
        result.append(
            dict(
                start=span[0], end=span[1], type=ENTITY_TYPES.index(entity["type"]) + 1, boxes=boxes
            )
        )
    return result


def _match_entities(entity, gold, c):
    device = entity["type_logits"].device
    e = entity["type_logits"].shape[0]
    types = torch.zeros(e, device=device, dtype=torch.long)
    if not gold:
        return [], types
    gt_type = torch.tensor([g["type"] for g in gold], device=device)
    starts = torch.tensor([g["start"] for g in gold], device=device)
    ends = torch.tensor([g["end"] - 1 for g in gold], device=device)
    with torch.no_grad():
        cost = -c["cost_entity_type"] * entity["type_logits"].float().log_softmax(-1)[:, gt_type]
        cost = cost - c["cost_entity_span"] * (
            entity["start_logits"].float().log_softmax(-1)[:, starts]
            + entity["end_logits"].float().log_softmax(-1)[:, ends]
        )
        assignment = _hungarian(cost)
    for slot, index in assignment:
        types[slot] = gt_type[index]
    return assignment, types


def _targets(output, gold, assignment):
    entity_count = len(output["entity"]["states"])
    images = output["num_images"]
    device = output["memory"].device
    result = [
        [torch.empty(0, 4, device=device, dtype=torch.float32) for _ in range(images)]
        for _ in range(entity_count)
    ]
    for slot, index in assignment:
        for image, boxes in enumerate(gold[index]["boxes"]):
            result[slot][image] = torch.tensor(boxes, device=device, dtype=torch.float32).reshape(
                -1, 4
            )
    return result


def _set_loss(stage, targets, c):
    logits = stage["logits"].float()
    boxes = stage["boxes"].float()
    e, m, k = logits.shape
    zero = logits.sum() * 0 + boxes.sum() * 0
    if not m:
        return zero
    counts = [len(box) for entity in targets for box in entity]
    maximum = max(counts, default=0)
    if maximum > k:
        raise ValueError(
            f"gold region count {maximum} exceeds auxiliary stage capacity {k}; increase query budget, never truncate gold"
        )
    groups = e * m
    flat_logits = logits.reshape(groups, k)
    flat_boxes = boxes.reshape(groups, k, 4)
    labels = torch.zeros_like(flat_logits)
    regression = zero + torch.zeros(groups, device=logits.device)
    chosen_groups = []
    chosen_sources = []
    chosen_gold = []
    if maximum:
        padded = boxes.new_zeros(groups, maximum, 4)
        for group, target in enumerate(box for entity in targets for box in entity):
            padded[group, : len(target)] = target
        with torch.no_grad():
            cost = (
                -c["cost_support"] * flat_logits.sigmoid()[..., None]
                + c["cost_l1"] * torch.cdist(flat_boxes, padded, p=1)
                + c["cost_giou"] * (1 - _pairwise_iou(flat_boxes, padded, giou=True))
            )
            cpu_cost = cost.cpu()
            for group, count in enumerate(counts):
                if count:
                    for source, dest in _hungarian(cpu_cost[group, :, :count]):
                        chosen_groups.append(group)
                        chosen_sources.append(source)
                        chosen_gold.append(padded[group, dest])
        groups_ix = torch.tensor(chosen_groups, device=logits.device, dtype=torch.long)
        sources_ix = torch.tensor(chosen_sources, device=logits.device, dtype=torch.long)
        target = torch.stack(chosen_gold)
        predicted = flat_boxes[groups_ix, sources_ix]
        labels[groups_ix, sources_ix] = 1
        l1 = (predicted - target).abs().sum(-1)
        # Aligned GIoU as independent 1x1 groups avoids a quadratic matched matrix.
        giou = 1 - _pairwise_iou(predicted[:, None, :], target[:, None, :], giou=True).reshape(-1)
        total = c["weight_l1"] * l1 + c["weight_giou"] * giou
        regression = regression.scatter_add(0, groups_ix, total) / torch.tensor(
            counts, device=logits.device
        ).clamp_min(1)
    element = F.binary_cross_entropy_with_logits(flat_logits, labels, reduction="none")
    positive = labels.bool()
    if c["support_loss_normalization"] == "positive_negative_balanced":
        positives = positive.sum(-1)
        negatives = k - positives
        classification = (element * positive).sum(-1) / positives.clamp_min(1)
        classification = classification + c["no_support_weight"] * (element * (~positive)).sum(
            -1
        ) / negatives.clamp_min(1)
    else:
        weights = torch.where(positive, 1.0, c["no_support_weight"])
        classification = (element * weights).sum(-1) / weights.sum(-1)
    return (c["weight_support"] * classification + regression).mean()


@torch.no_grad()
def _residual_targets(refined, targets, c):
    result = []
    for entity, images in enumerate(targets):
        per_image = []
        for image, gold in enumerate(images):
            retained = (
                refined["boxes"][entity, image][
                    refined["logits"][entity, image].float().sigmoid() >= c["residual_threshold"]
                ]
                .detach()
                .float()
            )
            overlaps = _pairwise_iou(retained, gold.detach().float())
            n, g = overlaps.shape
            covered = set()
            if n and g:
                eligible = overlaps > c["iou_threshold"]
                bonus = float(min(n, g) + 1)
                cost = torch.zeros(n, g + n, device=overlaps.device)
                cost[:, :g] = torch.where(eligible, -bonus - overlaps, bonus)
                covered = {
                    column
                    for row, column in _hungarian(cost)
                    if column < g and bool(eligible[row, column])
                }
            keep = [i for i in range(len(gold)) if i not in covered]
            per_image.append(gold[keep].detach().clone())
        result.append(per_image)
    return result


def auxiliary_loss(
    output: Mapping[str, Any],
    gold_entities: Sequence[Mapping[str, Any]],
    num_words: int,
    num_images: int,
) -> dict[str, Tensor]:
    """Return exactly {'entity', 'set'}; trainer applies its external lambdas.

    Entity Hungarian matching uses start/end/type predictions. Per-entity/image
    Hungarian region matching uses support/L1/GIoU. Unmatched entity slots get
    non-entity/no-support targets. Completion uses detached, strictly-IoU
    uncovered gold targets. No target influences the already computed memory.
    """
    gold = _validate_gold(output, gold_entities, num_words, num_images)
    c = output["config"]
    entity = output["entity"]
    assignment, types = _match_entities(entity, gold, c)
    weights = entity["type_logits"].new_ones(5).float()
    weights[0] = c["ner_negative_weight"]
    type_loss = F.cross_entropy(entity["type_logits"].float(), types, weight=weights)
    span_loss = entity["start_logits"].float().sum() * 0 + entity["end_logits"].float().sum() * 0
    if assignment:
        slots = torch.tensor([s for s, _ in assignment], device=types.device)
        starts = torch.tensor([gold[g]["start"] for _, g in assignment], device=types.device)
        ends = torch.tensor([gold[g]["end"] - 1 for _, g in assignment], device=types.device)
        span_loss = F.cross_entropy(
            entity["start_logits"][slots].float(), starts
        ) + F.cross_entropy(entity["end_logits"][slots].float(), ends)
    entity_loss = c["weight_entity_type"] * type_loss + c["weight_entity_span"] * span_loss
    targets = _targets(output, gold, assignment)
    components = {}
    for name in ("discovery", "refined", "final"):
        if name in output["stages"]:
            components[name] = _set_loss(output["stages"][name], targets, c)
    if "completion" in output["stages"]:
        residual = _residual_targets(output["stages"]["refined"], targets, c)
        components["completion"] = _set_loss(output["stages"]["completion"], residual, c)
    set_objective = sum(
        (c["loss_" + name] * loss for name, loss in components.items()), entity_loss * 0
    )
    return dict(entity=entity_loss, set=set_objective)
