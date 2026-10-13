"""Entity-conditioned set decoder over a square grid of visual patch features."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ENTITY_TYPES, entity_type_id, generalized_box_iou
from ..geometry import minimum_cost_assignment


class SetGroundingDecoder(nn.Module):
    """Predict several boxes and an explicit visual-presence score per entity/image."""

    def __init__(self, input_dim=768, hidden=256, queries=128, heads=8, layers=2):
        super().__init__()
        self.vision = nn.Linear(input_dim, hidden)
        self.mention = nn.Linear(input_dim, hidden)
        self.entity_type = nn.Embedding(len(ENTITY_TYPES), hidden)
        self.position = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.query = nn.Embedding(queries, hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.objectness = nn.Linear(hidden, 1)
        self.box = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.presence = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, patches: Tensor, mention: Tensor, entity_types: Tensor) -> dict[str, Tensor]:
        """Inputs: [batch, patches, dim], [batch, dim], and [batch]."""
        batch, length, _ = patches.shape
        side = int(length ** 0.5)
        if side == 0 or side * side != length:
            raise ValueError("SD requires a nonempty square patch grid without a CLS token")
        axis = (torch.arange(side, device=patches.device, dtype=patches.dtype) + 0.5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((xx, yy), dim=-1).reshape(length, 2)
        context = self.mention(mention) + self.entity_type(entity_types)
        memory = self.vision(patches) + self.position(positions)[None] + context[:, None]
        queries = self.query.weight[None].expand(batch, -1, -1) + context[:, None]
        hidden = self.decoder(queries, memory)
        center_size = self.box(hidden).sigmoid()
        center, size = center_size[..., :2], center_size[..., 2:].clamp(min=1e-4)
        boxes = torch.cat((center - size / 2, center + size / 2), dim=-1).clamp(0, 1)
        return {"logits": self.objectness(hidden).squeeze(-1), "boxes": boxes,
                "presence": self.presence(torch.cat((memory.mean(1), context), -1)).squeeze(-1)}


def set_loss(output: Mapping[str, Tensor], targets: Sequence[Tensor]) -> Tensor:
    """Hungarian matching with BCE, L1, GIoU, and explicit empty-target supervision.

    Each target contains all observed normalized xyxy boxes for one entity/image.
    An empty target means a known absence; omit pairs with unknown supervision.
    """
    if not targets or len(targets) != len(output["logits"]):
        raise ValueError("One box tensor is required for each entity/image pair")
    total = output["logits"].sum() * 0
    for index, target in enumerate(targets):
        boxes, logits = output["boxes"][index], output["logits"][index]
        target = torch.as_tensor(target, device=boxes.device, dtype=boxes.dtype).reshape(-1, 4)
        if len(target) > len(boxes):
            raise ValueError("Reference boxes exceed SD query capacity; increase queries")
        if (not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any()
                or (target[:, 2:] <= target[:, :2]).any()):
            raise ValueError("SD targets require valid normalized xyxy boxes")
        expected = torch.zeros_like(logits)
        regression = boxes.sum() * 0
        if len(target):
            cost = (-logits.sigmoid()[:, None] + 5 * torch.cdist(boxes, target, p=1)
                    - 2 * generalized_box_iou(boxes, target))
            if not torch.isfinite(cost).all():
                raise FloatingPointError("Nonfinite Hungarian matching cost")
            pairs = minimum_cost_assignment(cost)
            rows, columns = zip(*pairs)
            rows = torch.as_tensor(rows, device=boxes.device, dtype=torch.long)
            columns = torch.as_tensor(columns, device=boxes.device, dtype=torch.long)
            expected[rows] = 1
            matched, reference = boxes[rows], target[columns]
            regression = (5 * F.l1_loss(matched, reference)
                          + 2 * (1 - generalized_box_iou(matched, reference).diag()).mean())
        weights = torch.where(expected.bool(), 1.0, 0.1)
        classification = F.binary_cross_entropy_with_logits(logits, expected, weight=weights)
        presence = F.binary_cross_entropy_with_logits(
            output["presence"][index], logits.new_tensor(float(len(target) > 0)))
        total = total + classification + regression + presence
    return total / len(targets)


def decode_sd(output: Mapping[str, Tensor], images: Sequence[Mapping], threshold=0.5) -> list[list[dict]]:
    """Map each entity/image pair to original-image pixel boxes."""
    if len(images) != len(output["boxes"]):
        raise ValueError("One image descriptor is required per SD output row")
    scores = (output["logits"].sigmoid() * output["presence"].sigmoid()[:, None]).detach().cpu()
    boxes = output["boxes"].detach().cpu()
    regions = []
    for image, row_boxes, row_scores in zip(images, boxes, scores):
        width, height = image["width"], image["height"]
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        scale = row_boxes.new_tensor([width, height, width, height])
        current = [{"image_id": str(image["image_id"]), "bbox": (box * scale).tolist(),
                    "score": float(score)} for box, score in zip(row_boxes, row_scores)
                   if score >= threshold and bool((box[2:] > box[:2]).all())]
        regions.append(sorted(current, key=lambda region: -region["score"]))
    return regions


@torch.inference_mode()
def ground_sd(model: SetGroundingDecoder, entities: Sequence[Mapping], text_features: Tensor,
              image_features: Mapping[str, Tensor], images: Sequence[Mapping], threshold=0.5) -> list[dict]:
    """Attach regions to NER spans using one word feature per original token.

    Image features are square ViT patch grids keyed by image_id. The text and
    image feature dimensions must match the SD input dimension.
    """
    model.eval()
    device = next(model.parameters()).device
    results = []
    for entity in entities:
        type_id = entity_type_id(entity, len(text_features))
        start, end = entity["token_span"]
        mention = text_features[start:end].mean(0)
        regions = []
        for image in images:
            patches = image_features[str(image["image_id"])]
            output = model(patches[None].to(device), mention[None].to(device),
                           torch.tensor([type_id], device=device))
            regions.extend(decode_sd(output, [image], threshold)[0])
        results.append({**entity, "regions": sorted(regions, key=lambda region: -region["score"])})
    return results
