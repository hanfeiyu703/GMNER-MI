# ELER Core

Direct-SFT with ELER for Qwen2.5-VL, together with SD / EVG adapter modules.

## Data

GMNER-MI and GMNER-MI-Plus pair tokenized social-media text with multiple images. Each entity has a type, a token span, and image regions. Empty region lists represent entities without visible evidence.

Use one JSON object per line in `train.jsonl`:

```json
{"sample_id":"example_1","tokens":["Avery","visits","Harbor"],"images":[{"image_id":"img_1","path":"images/img_1.jpg","width":640,"height":480}],"entities":[{"token_span":[0,1],"type":"PER","regions":[{"image_id":"img_1","box":[40,30,180,240]}]}]}
```

Token spans use an exclusive end index; input boxes are pixel coordinates `[x1,y1,x2,y2]`. Entity types are `PER`, `LOC`, `ORG`, and `MIS` (`MISC` is also accepted). Image paths are relative to `--image-root`. The original dataset JSONL format is also supported.

## Training

Python 3.10+ and a CUDA GPU are recommended for the 7B model.

```bash
pip install -r requirements.txt
python -m eler.train --config configs/train.json \
  --data data/mi/train.jsonl --image-root data/mi \
  --output runs/eler --epochs 1
```

Use `--model models/Qwen2.5-VL-7B-Instruct` for local model weights. The configuration controls batch size, image resolution, query counts, and learning rates. Each epoch saves LoRA, ELER, and processor files under the output directory. Set `--variant baseline` to run Direct-SFT alone.

For a small synthetic example:

```bash
python examples/make_demo.py
python -m eler.train --config configs/train.json \
  --data data/demo/train.jsonl --image-root data/demo --output runs/demo
```

## SD / EVG

`eler/adapters/sd.py` provides an entity-conditioned set decoder and multi-region loss. `eler/adapters/evg.py` provides a BERT / VinVL ROI grounding head. Both adapt entity predictions into image-region outputs. EVG expects extracted 2048-dimensional ROI features.

```bash
python examples/adapt_heads.py
```

This example uses synthetic features to demonstrate the adapter interfaces.
