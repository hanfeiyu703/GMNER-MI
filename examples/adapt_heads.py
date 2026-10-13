"""Small CPU example of the SD/EVG interfaces; random features only exercise shapes.

For model training, replace these random tensors with word/ViT features for SD
and actual VinVL 2048-D ROI features for EVG. VinVL extraction is separate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from transformers import BertConfig, BertModel, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eler.adapters import (EVG, SetGroundingDecoder, candidate_targets, collate_evg,
                           decode_evg, decode_sd, encode_evg, evg_loss, ground_evg,
                           ground_sd, normalized_boxes, set_loss)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    tokens = ["Ada", "visited", "Paris"]
    entities = [{"token_span": [0, 1], "type": "PER"},
                {"token_span": [2, 3], "type": "LOC"}]
    images = [{"image_id": "image_a", "width": 100, "height": 80},
              {"image_id": "image_b", "width": 120, "height": 90}]

    sd = SetGroundingDecoder(input_dim=32, hidden=32, queries=4, heads=4, layers=1)
    text_features = torch.randn(len(tokens), 32)
    patch_features = {image["image_id"]: torch.randn(4, 32) for image in images}
    output = sd(torch.stack(list(patch_features.values())),
                text_features[:1].expand(2, -1), torch.tensor([0, 0]))
    targets = [normalized_boxes([[10, 12, 45, 60], [60, 12, 85, 45]], 100, 80),
               torch.empty(0, 4)]
    loss_sd = set_loss(output, targets)
    loss_sd.backward()
    assert torch.isfinite(loss_sd) and sd.presence[-1].weight.grad is not None
    assert decode_sd(output, images, threshold=1.0) == [[], []]
    sd_predictions = ground_sd(sd, entities, text_features, patch_features, images, threshold=0.0)

    with TemporaryDirectory() as folder:
        vocabulary = Path(folder) / "vocab.txt"
        vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nada\nvisited\nparis\nper\nloc\norg\nmis\n")
        tokenizer = BertTokenizer(vocab_file=str(vocabulary), model_max_length=64)
        bert = BertModel(BertConfig(vocab_size=12, hidden_size=32, num_hidden_layers=1,
                                   num_attention_heads=4, intermediate_size=64,
                                   max_position_embeddings=64))
        evg = EVG(bert, heads=4)
        candidates = [{"image_id": "image_a", "bbox": [10, 12, 45, 60]},
                      {"image_id": "image_a", "bbox": [11, 12, 46, 60]},
                      {"image_id": "image_b", "bbox": [10, 12, 45, 60]}]
        roi_features = torch.randn(len(candidates), 2048)
        positives = candidate_targets(candidates, [candidates[0]])
        assert positives.tolist() == [1.0, 1.0, 0.0]
        rows = [{"encoding": encode_evg(tokenizer, tokens, entity, max_length=64),
                 "features": roi_features,
                 "targets": positives if index == 0 else candidate_targets(candidates, [])}
                for index, entity in enumerate(entities)]
        batch = collate_evg(rows, tokenizer.pad_token_id)
        target_rois = batch.pop("targets")
        logits = evg(**batch)
        loss_evg = evg_loss(logits, target_rois, batch["proposal_mask"])
        loss_evg.backward()
        assert torch.isfinite(loss_evg) and evg.cross_attention.in_proj_weight.grad is not None
        assert decode_evg(logits[0], candidates, threshold=1.0) == []
        assert decode_evg(torch.tensor([-5.0, -5.0, 5.0]), candidates)[0]["image_id"] == "image_b"
        evg_predictions = ground_evg(evg, tokenizer, tokens, entities, roi_features, candidates)
        assert all(entity["regions"] == [] for entity in
                   ground_evg(evg, tokenizer, tokens, entities, torch.empty(0, 2048), []))

    print(json.dumps({"sd_loss": round(float(loss_sd.detach()), 4),
                      "evg_loss": round(float(loss_evg.detach()), 4),
                      "sd_entities": len(sd_predictions), "evg_entities": len(evg_predictions)}, indent=2))


if __name__ == "__main__":
    main()
