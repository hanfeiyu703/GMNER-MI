"""Single-device Direct-SFT and ELER training with response-only supervision."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

from .data import collate_items, read_training_items

DEFAULT_CONFIG = dict(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct", model_revision=None,
    variant="full", mode="full", train_file="data/train.jsonl", image_root="data",
    output_dir="runs/eler", device="cuda:0", compute_dtype="bfloat16",
    attn_implementation="sdpa", min_pixels=50176, max_pixels=200704,
    max_seq_length=8192, epochs=1, microbatch_size=1, gradient_accumulation_steps=4,
    learning_rate=5e-5, module_learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.05,
    lora_gradient_clip=1.0, module_gradient_clip=1.0, gradient_checkpointing=True,
    lora_rank=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    freeze_vision=True, seed=42, entity_types=["PER", "ORG", "LOC", "MISC"],
    prewarm_epochs=0.25, entity_aux_weight=0.01, set_aux_weight=0.05,
    hidden_dim=256, entity_queries=16, discovery_queries=128, completion_queries=128,
    attention_heads=8, reader_layers=2, feedforward_dim=1024, max_images=16,
    mc_samples=8, dropout=0.1, support_dropout=0.1, no_support_weight=0.1,
    uncertainty_eta=0.5, gate_init=0.05,
)


def read_config(path):
    config = dict(DEFAULT_CONFIG)
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a JSON object")
    config.update(supplied)
    return config


def validate_config(config):
    if config["variant"] not in {"baseline", "full"}:
        raise ValueError("variant must be baseline or full")
    for key in ("epochs", "microbatch_size", "gradient_accumulation_steps", "max_seq_length"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "module_learning_rate", "lora_gradient_clip", "module_gradient_clip"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("prewarm_epochs", "entity_aux_weight", "set_aux_weight", "weight_decay"):
        if not math.isfinite(float(config[key])) or float(config[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= float(config["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratio must be in [0,1)")
    if config["compute_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("compute_dtype must be bfloat16 or float32")


def response_only_post_loss(model, batch, *, prewarm=False):
    """Average supervised tokens within each post, then average the posts."""
    import torch
    labels = batch["labels"]
    positions = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=False).flatten()
    if not positions.numel():
        raise ValueError("No supervised response tokens in this batch")
    result = model(**batch["model_inputs"], public_metadata=batch["public_metadata"],
                   detach_evidence_inputs=prewarm, use_cache=False, logits_to_keep=positions)
    if result.logits.shape[1] != positions.numel():
        raise ValueError("Model must honor logits_to_keep positions")
    targets = labels[:, positions + 1]
    valid = targets != -100
    lengths = valid.sum(1)
    if not bool((lengths > 0).all()):
        raise ValueError("A post has no supervised response tokens")
    losses = torch.nn.functional.cross_entropy(result.logits.float().transpose(1, 2),
                                               targets, ignore_index=-100, reduction="none")
    return ((losses * valid).sum(1) / lengths).mean(), int(lengths.sum()), result


def parameter_groups(model):
    lora, evidence = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lora if "lora_" in name else evidence).append(parameter)
    if not lora and not evidence:
        raise ValueError("Model has no trainable parameters")
    return lora, evidence


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_training(config):
    import torch
    from .generative import load_model
    validate_config(config)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose a CUDA host or use device=cpu with a small model")
    if device.type == "cuda" and config["compute_dtype"] == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This CUDA device does not support bfloat16; use float32")
    rows = read_training_items(config["train_file"], config["image_root"])
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    processor, model, _ = load_model(config, str(device))
    lora, evidence = parameter_groups(model)
    groups = []
    if lora:
        groups.append(dict(params=lora, lr=config["learning_rate"]))
    if evidence:
        groups.append(dict(params=evidence, lr=config["module_learning_rate"]))
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    microbatch = config["microbatch_size"]
    accumulation = config["gradient_accumulation_steps"]
    nominal_posts = microbatch * accumulation
    batches = math.ceil(len(rows) / microbatch)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    warmup = int(total_updates * config["warmup_ratio"])
    def lr_scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    prewarm_updates = math.ceil(config["prewarm_epochs"] * updates_per_epoch)
    write_json(output / "config.json", config)
    started, step = time.monotonic(), 0
    model.train()
    for epoch in range(config["epochs"]):
        order = list(range(len(rows)))
        random.Random(config["seed"] + epoch).shuffle(order)
        epoch_loss, epoch_posts, window_posts = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for number, begin in enumerate(range(0, len(order), microbatch), 1):
            prewarm = config["variant"] == "full" and step < prewarm_updates
            model.set_prewarm(prewarm)
            items = [rows[i] for i in order[begin:begin + microbatch]]
            batch = collate_items(items, processor, config)
            batch["model_inputs"] = {key: value.to(device) for key, value in batch["model_inputs"].items()}
            batch["labels"] = batch["labels"].to(device)
            context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if config["compute_dtype"] == "bfloat16" else nullcontext()
            with context:
                ce, _, result = response_only_post_loss(model, batch, prewarm=prewarm)
                auxiliary = model.auxiliary_loss(result.evidence_outputs, batch["gold_entities"], batch["public_metadata"])
                loss = ce + config["entity_aux_weight"] * auxiliary.get("entity", ce.new_zeros(()))
                loss = loss + config["set_aux_weight"] * auxiliary.get("set", ce.new_zeros(()))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            # Each post receives equal weight, including the final short group.
            (loss * len(items) / nominal_posts).backward()
            window_posts += len(items)
            epoch_posts += len(items)
            epoch_loss += float(loss.detach()) * len(items)
            if number % accumulation == 0 or number == batches:
                if window_posts != nominal_posts:
                    for parameter in lora + evidence:
                        if parameter.grad is not None:
                            parameter.grad.mul_(nominal_posts / window_posts)
                for parameters, limit in ((lora, config["lora_gradient_clip"]), (evidence, config["module_gradient_clip"])):
                    if parameters:
                        norm = torch.nn.utils.clip_grad_norm_(parameters, limit)
                        if not bool(torch.isfinite(norm)):
                            raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_posts = 0
                print(json.dumps(dict(epoch=epoch + 1, batch=number, batches=batches, step=step,
                                      loss=round(float(loss.detach()), 6), prewarm=prewarm)), flush=True)
            del batch, result, auxiliary, ce, loss
        checkpoint = output / f"epoch_{epoch + 1:02d}"
        checkpoint.mkdir()
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
        record = dict(epoch=epoch + 1, steps=step, posts=epoch_posts, mean_loss=epoch_loss / epoch_posts,
                      elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(checkpoint / "training.json", record)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    return dict(epochs=config["epochs"], steps=step, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("baseline", "full"))
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--image-root")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = read_config(args.config)
    for argument, key in (("variant", "variant"), ("model", "model_id"), ("data", "train_file"), ("image_root", "image_root"),
                          ("output", "output_dir"), ("epochs", "epochs"), ("device", "device")):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    print(json.dumps(run_training(config), indent=2), flush=True)


if __name__ == "__main__":
    main()
