"""Shared entity and box helpers for optional grounding heads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ..geometry import box_iou, generalized_box_iou


ENTITY_TYPES = ("PER", "LOC", "ORG", "MIS")
TYPE_IDS = {name: index for index, name in enumerate(ENTITY_TYPES)}


def entity_type_id(entity: Mapping, token_count: int) -> int:
    """Validate a half-open word-token span and return its type ID."""
    span = entity.get("token_span")
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= token_count):
        raise ValueError("Entities require a valid half-open token_span")
    name = entity.get("type")
    name = "MIS" if name == "MISC" else name
    if name not in TYPE_IDS:
        raise ValueError("Entity type must be PER, LOC, ORG, or MIS")
    return TYPE_IDS[name]


def normalized_boxes(boxes: Sequence, width: float, height: float) -> Tensor:
    """Convert original-image pixel xyxy boxes to normalized xyxy."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    value = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    scale = value.new_tensor([width, height, width, height])
    if (not torch.isfinite(value).all() or (value < 0).any()
            or (value > scale).any() or (value[:, 2:] <= value[:, :2]).any()):
        raise ValueError("Boxes must be valid original-image pixel xyxy")
    return value / scale
