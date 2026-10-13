"""Create a small synthetic training dataset in the documented JSONL format."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    root = Path(args.output)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate(["steelblue", "coral"]):
        image = Image.new("RGB", (112, 112), "white")
        ImageDraw.Draw(image).rectangle((16, 16, 96, 96), fill=color)
        image.save(images / f"image_{index}.png")
    rows = [
        {
            "sample_id": "demo_1",
            "tokens": ["Avery", "visits", "Harbor"],
            "images": [
                {"image_id": "image_0", "path": "images/image_0.png", "width": 112, "height": 112},
                {"image_id": "image_1", "path": "images/image_1.png", "width": 112, "height": 112},
            ],
            "entities": [
                {"token_span": [0, 1], "type": "PER", "regions": [
                    {"image_id": "image_0", "box": [16, 16, 96, 96]},
                    {"image_id": "image_1", "box": [16, 16, 96, 96]},
                ]},
                {"token_span": [2, 3], "type": "LOC", "regions": []},
            ],
        },
        {
            "sample_id": "demo_2",
            "tokens": ["A", "quiet", "afternoon"],
            "images": [
                {"image_id": "image_0", "path": "images/image_0.png", "width": 112, "height": 112},
            ],
            "entities": [],
        },
    ]
    with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(root / "train.jsonl")


if __name__ == "__main__":
    main()
