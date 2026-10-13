"""BERT entity-enhanced text and VinVL ROI cross-modal grounding."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ENTITY_TYPES, box_iou, entity_type_id


def encode_evg(tokenizer, tokens: Sequence[str], entity: Mapping, max_length=512) -> dict[str, Tensor]:
    """Encode [CLS] sentence [SEP] mention [SEP] entity-type [SEP]."""
    type_id = entity_type_id(entity, len(tokens))
    start, end = entity["token_span"]
    texts = (" ".join(tokens), " ".join(tokens[start:end]), ENTITY_TYPES[type_id])
    pieces = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    ids = [tokenizer.cls_token_id] + pieces[0] + [tokenizer.sep_token_id]
    types = [0] * len(ids)
    ids += pieces[1] + [tokenizer.sep_token_id] + pieces[2] + [tokenizer.sep_token_id]
    types += [1] * (len(ids) - len(types))
    if len(ids) > min(max_length, tokenizer.model_max_length):
        raise ValueError("Entity-enhanced text exceeds BERT capacity")
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(types, dtype=torch.long)}


class EVG(nn.Module):
    """Cross-modal transformer CMT(V,T,T) with shared sigmoid ROI scoring.

    Pass a pretrained BertModel for normal use. ROI inputs are the actual
    2048-dimensional features extracted by VinVL, with candidate coordinates
    maintained separately in their original images.
    """

    def __init__(self, bert: nn.Module, heads=12):
        super().__init__()
        self.bert = bert
        dim = bert.config.hidden_size
        self.projection = nn.Linear(2048, dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm_attention = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                         nn.Dropout(0.1), nn.Linear(dim * 4, dim))
        self.norm_output = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor,
                features: Tensor, proposal_mask: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != 2048:
            raise ValueError("EVG requires [batch, candidates, 2048] VinVL ROI features")
        if not proposal_mask.bool().any(-1).all():
            raise ValueError("Each EVG row requires at least one ROI candidate")
        text = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                         attention_mask=attention_mask).last_hidden_state
        visual = self.projection(features)
        mixed, _ = self.cross_attention(visual, text, text,
                                       key_padding_mask=~attention_mask.bool(), need_weights=False)
        hidden = self.norm_attention(visual + self.dropout(mixed))
        hidden = self.norm_output(hidden + self.dropout(self.feedforward(hidden)))
        return self.score(hidden).squeeze(-1).masked_fill(~proposal_mask.bool(), -1e4)


def collate_evg(rows: Sequence[Mapping], pad_id: int) -> dict[str, Tensor]:
    """Pad encoded entity requests and their ROI features into a batch."""
    if not rows:
        raise ValueError("EVG batch must contain at least one entity request")
    max_text = max(len(row["encoding"]["input_ids"]) for row in rows)
    max_regions = max(len(row["features"]) for row in rows)
    batch = len(rows)
    output = {"input_ids": torch.full((batch, max_text), pad_id, dtype=torch.long),
              "token_type_ids": torch.zeros(batch, max_text, dtype=torch.long),
              "attention_mask": torch.zeros(batch, max_text, dtype=torch.long),
              "features": torch.zeros(batch, max_regions, 2048),
              "proposal_mask": torch.zeros(batch, max_regions, dtype=torch.bool)}
    with_targets = all("targets" in row for row in rows)
    if any("targets" in row for row in rows) and not with_targets:
        raise ValueError("EVG batch cannot mix labeled and unlabeled requests")
    if with_targets:
        output["targets"] = torch.zeros(batch, max_regions)
    for index, row in enumerate(rows):
        text_length, regions = len(row["encoding"]["input_ids"]), len(row["features"])
        if regions == 0 or row["features"].shape != (regions, 2048):
            raise ValueError("Each request requires nonempty 2048-D VinVL ROI features")
        for name in ("input_ids", "token_type_ids"):
            output[name][index, :text_length] = row["encoding"][name]
        output["attention_mask"][index, :text_length] = 1
        output["features"][index, :regions] = row["features"]
        output["proposal_mask"][index, :regions] = True
        if with_targets:
            output["targets"][index, :regions] = row["targets"]
    return output


def candidate_targets(candidates: Sequence[Mapping], regions: Sequence[Mapping], iou_threshold=0.5) -> Tensor:
    """Supervise all matching ROIs; compare boxes only within the same image.

    Empty regions represent a known absence and produce all-negative targets.
    Candidate boxes are detector outputs, never annotation-derived proposals.
    """
    if regions is None:
        raise ValueError("Unknown region annotations cannot become negative targets")
    values = torch.zeros(len(candidates), dtype=torch.float32)
    for index, candidate in enumerate(candidates):
        references = [region["bbox"] for region in regions
                      if str(region["image_id"]) == str(candidate["image_id"])]
        if references:
            overlap = box_iou(torch.tensor([candidate["bbox"]], dtype=torch.float32),
                              torch.tensor(references, dtype=torch.float32))
            values[index] = float(bool((overlap > iou_threshold).any()))
    return values


def evg_loss(logits: Tensor, targets: Tensor, proposal_mask: Tensor) -> Tensor:
    """Multi-positive BCE, averaged within each entity then over the batch."""
    if logits.shape != targets.shape or logits.shape != proposal_mask.shape:
        raise ValueError("EVG logits, targets, and candidate mask must share a shape")
    mask = proposal_mask.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    return ((loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def decode_evg(logits: Tensor, candidates: Sequence[Mapping], threshold=0.5) -> list[dict]:
    """Return the highest-scoring actual ROI, or empty regions for no visual match."""
    if not candidates:
        return []
    if logits.ndim != 1 or len(logits) != len(candidates):
        raise ValueError("One EVG logit is required per candidate")
    scores = logits.detach().sigmoid().cpu()
    best = int(scores.argmax())
    if float(scores[best]) <= threshold:
        return []
    region = candidates[best]
    return [{"image_id": str(region["image_id"]), "bbox": list(region["bbox"]),
             "score": float(scores[best])}]


@torch.inference_mode()
def ground_evg(model: EVG, tokenizer, tokens: Sequence[str], entities: Sequence[Mapping],
               roi_features: Tensor, candidates: Sequence[Mapping], threshold=0.5) -> list[dict]:
    """Attach top-one ROI regions to NER entities, preserving original image IDs."""
    if roi_features.shape != (len(candidates), 2048):
        raise ValueError("ROI features and candidate metadata must align")
    model.eval()
    results = []
    device = next(model.parameters()).device
    for entity in entities:
        encoding = encode_evg(tokenizer, tokens, entity, model.bert.config.max_position_embeddings)
        if not candidates:
            regions = []
        else:
            batch = collate_evg([{"encoding": encoding, "features": roi_features}], tokenizer.pad_token_id)
            logits = model(**{name: value.to(device) for name, value in batch.items()})[0]
            regions = decode_evg(logits, candidates, threshold)
        results.append({**entity, "regions": regions})
    return results
