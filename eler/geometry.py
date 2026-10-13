"""Box geometry and exact, dependency-free bipartite assignment.

Boxes use normalized ``(x_min, y_min, x_max, y_max)`` coordinates. Matching
decisions deliberately operate on detached CPU values, while box geometry
keeps PyTorch autograd intact for the regression objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _box_matrix(boxes: Tensor) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected [N, 4] boxes; received {tuple(boxes.shape)}")
    # BF16/FP16 epsilon is far too large for normalized small-region areas.
    # Upcasting is differentiable and also makes integer fixtures safe.
    if not boxes.is_floating_point() or boxes.dtype in (torch.float16, torch.bfloat16):
        boxes = boxes.float()
    return boxes


def box_area(boxes: Tensor) -> Tensor:
    boxes = _box_matrix(boxes)
    side = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return side[:, 0] * side[:, 1]


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU, including well-defined empty matrices and zero areas."""
    boxes1, boxes2 = _box_matrix(boxes1), _box_matrix(boxes2)
    lower = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    upper = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    side = (upper - lower).clamp(min=0)
    intersection = side[..., 0] * side[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - intersection
    epsilon = torch.finfo(union.dtype).tiny
    return intersection / union.clamp(min=epsilon)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise generalized IoU (Rezatofighi et al., 2019)."""
    boxes1, boxes2 = _box_matrix(boxes1), _box_matrix(boxes2)
    lower = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    upper = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    side = (upper - lower).clamp(min=0)
    intersection = side[..., 0] * side[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - intersection
    enclosure_lower = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosure_upper = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosure_side = (enclosure_upper - enclosure_lower).clamp(min=0)
    enclosure = enclosure_side[..., 0] * enclosure_side[..., 1]
    epsilon = torch.finfo(union.dtype).tiny
    return intersection / union.clamp(min=epsilon) - (enclosure - union) / enclosure.clamp(
        min=epsilon
    )


def minimum_cost_assignment(cost: Tensor | Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Exact rectangular Hungarian assignment without SciPy.

    Returns ``min(rows, columns)`` pairs, sorted by row. Every vertex in the
    smaller partition is assigned; use dummy columns for optional matches.
    Empty dimensions return no pairs. Finite negative costs are supported.
    """
    if isinstance(cost, Tensor):
        if cost.ndim != 2:
            raise ValueError("Assignment cost must be a matrix")
        n_rows, n_cols = cost.shape
        values = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    else:
        values = [list(map(float, row)) for row in cost]
        n_rows = len(values)
        n_cols = len(values[0]) if n_rows else 0
        if any(len(row) != n_cols for row in values):
            raise ValueError("Assignment matrix must be rectangular")
    if n_rows == 0 or n_cols == 0:
        return []
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("Assignment costs must be finite")
    transpose = n_rows > n_cols
    if transpose:
        values = [list(row) for row in zip(*values)]
    rows, columns = len(values), len(values[0])
    # Augmenting-path Hungarian algorithm, O(n^2 m).
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    assigned_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row in range(1, rows + 1):
        assigned_row[0] = row
        column0 = 0
        min_slack = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = assigned_row[column0]
            delta, column1 = math.inf, 0
            for column in range(1, columns + 1):
                if not used[column]:
                    reduced = values[row0 - 1][column - 1] - u[row0] - v[column]
                    if reduced < min_slack[column]:
                        min_slack[column], predecessor[column] = reduced, column0
                    if min_slack[column] < delta:
                        delta, column1 = min_slack[column], column
            for column in range(columns + 1):
                if used[column]:
                    u[assigned_row[column]] += delta
                    v[column] -= delta
                else:
                    min_slack[column] -= delta
            column0 = column1
            if assigned_row[column0] == 0:
                break
        while column0:
            column1 = predecessor[column0]
            assigned_row[column0] = assigned_row[column1]
            column0 = column1
    result = []
    for column in range(1, columns + 1):
        if assigned_row[column]:
            pair = (assigned_row[column] - 1, column - 1)
            result.append((pair[1], pair[0]) if transpose else pair)
    return sorted(result)


def maximum_bipartite_matching(quality: Tensor, eligible: Tensor) -> list[tuple[int, int]]:
    """Maximize valid match count first, then sum of bounded quality scores.

    ``quality`` must lie in [0, 1]; ``eligible`` marks allowed edges. Dummy
    columns allow every row to remain unmatched. This avoids the greedy
    assignment failure where one strong match blocks two compatible matches.
    """
    if quality.ndim != 2 or quality.shape != eligible.shape:
        raise ValueError("Matching quality and eligible masks must be equal-sized matrices")
    rows, columns = quality.shape
    if not rows or not columns:
        return []
    quality = quality.detach().to(device="cpu", dtype=torch.float64)
    eligible = eligible.detach().to(device="cpu", dtype=torch.bool)
    if not torch.isfinite(quality).all() or (quality < 0).any() or (quality > 1).any():
        raise ValueError("Matching quality must be finite and in [0, 1]")
    bonus = float(min(rows, columns) + 1)
    cost = torch.zeros((rows, columns + rows), dtype=torch.float64)
    cost[:, :columns] = torch.where(eligible, -bonus - quality, bonus)
    return [
        (row, column)
        for row, column in minimum_cost_assignment(cost)
        if column < columns and bool(eligible[row, column])
    ]


def maximum_iou_matching(
    boxes1: Tensor, boxes2: Tensor, threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Exact matching under the paper's strict ``IoU > threshold`` criterion."""
    overlaps = box_iou(boxes1.detach(), boxes2.detach())
    return maximum_bipartite_matching(overlaps, overlaps > threshold)
