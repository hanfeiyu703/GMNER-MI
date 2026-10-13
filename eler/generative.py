"""Keep native Qwen generation/cache and add public-only evidence at its LM head.

Hooks observe the complete native core output before logits_to_keep selects
answer positions. They never replace the Qwen core or its vocabulary head.
The request context lives only for a forward/generate call; no evidence is
reused between requests. Checkpointed decoder layers never execute these hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


class EvidenceReadout(nn.Module):
    def __init__(self, hidden_size, memory_dim=256, heads=8, gate_init=0.05):
        super().__init__()
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must lie strictly between zero and one")
        self.query = nn.Linear(hidden_size, memory_dim)
        self.attention = nn.MultiheadAttention(memory_dim, heads, batch_first=True)
        self.output = nn.Linear(memory_dim, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1 - gate_init))))
        nn.init.zeros_(self.output.weight)

    def forward(self, states, memories, active, biases=None):
        # Skip inactive prompt positions rather than evaluating a future-aware
        # memory read there. This also reduces training work and peak memory.
        rows = []
        for b, memory in enumerate(memories):
            indices = active[b].nonzero(as_tuple=False).flatten()
            row = states[b]
            if indices.numel():
                q = self.query(row[indices].float())[None]
                bias = (
                    None
                    if biases is None or biases[b] is None
                    else biases[b].float()[None].expand(len(indices), -1)
                )
                context, _ = self.attention(
                    q,
                    memory[None].float(),
                    memory[None].float(),
                    attn_mask=bias,
                    need_weights=False,
                )
                delta = self.output(context[0]) * self.gate_logit.sigmoid()
                row = row.index_add(0, indices, delta.to(dtype=row.dtype))
            rows.append(row)
        return torch.stack(rows)


class GenerativeQwen(nn.Module):
    def __init__(self, generator, config, evidence=None):
        super().__init__()
        self.generator = generator
        self.run_config = dict(config)
        self.variant = config.get("variant", "baseline")
        if self.variant not in ("baseline", "full"):
            raise ValueError("variant must be baseline or full")
        native = generator.get_base_model() if hasattr(generator, "get_base_model") else generator
        self.hidden_size = int(native.config.text_config.hidden_size)
        self.merge_size = int(native.config.vision_config.spatial_merge_size)
        self._context = None
        self._rng_step = 0
        self._prewarm = False
        self._handles = []
        if self.variant != "baseline":
            if evidence is None:
                from .evidence import GenerativeEvidence

                evidence = GenerativeEvidence(config, backbone_dim=self.hidden_size)
            self.evidence = evidence
            self.readout = EvidenceReadout(
                self.hidden_size,
                int(config.get("hidden_dim", 256)),
                int(config.get("attention_heads", 8)),
                float(config.get("gate_init", 0.05)),
            )
            self._handles = [
                native.model.register_forward_hook(self._observe_core),
                native.lm_head.register_forward_pre_hook(self._enhance_head),
            ]
        else:
            self.evidence = None
            self.readout = None

    @property
    def device(self):
        return next(self.generator.parameters()).device

    @property
    def config(self):
        return self.generator.config

    @property
    def generation_config(self):
        return self.generator.generation_config

    def set_prewarm(self, enabled):
        self._prewarm = bool(enabled)
        if self.readout is not None:
            self.readout.output.weight.requires_grad_(not enabled)

    def lora_parameters(self):
        return [
            p
            for name, p in self.generator.named_parameters()
            if p.requires_grad and "lora_" in name
        ]

    def module_parameters(self):
        # Include output projection while frozen for prewarm: it must already
        # exist in the optimizer when subsequently unfrozen.
        if self.evidence is None:
            return []
        return list(self.evidence.parameters()) + list(self.readout.parameters())

    @contextmanager
    def _request(self, metadata, mode, keep=0, detach=False):
        if self._context is not None:
            raise RuntimeError("Concurrent/reentrant requests on one model are unsupported")
        if self.evidence is not None and not metadata:
            raise ValueError("Generative evidence requires public token metadata")
        names = "|".join(str(m.get("sample_id", "")) for m in metadata or [])
        salt = int.from_bytes(hashlib.sha256(names.encode()).digest()[:4], "little")
        seed = (
            int(self.run_config.get("seed", 1234)) + salt + (self._rng_step if self.training else 0)
        ) % (2**31)
        self._context = dict(
            metadata=metadata or [],
            mode=mode,
            keep=keep,
            detach=detach or self._prewarm,
            outputs=None,
            core_length=None,
            seed=seed,
        )
        if self.training:
            self._rng_step += 1
        try:
            yield self._context
        finally:
            self._context = None

    def _observe_core(self, _module, _args, output):
        ctx = self._context
        if ctx is None:
            raise RuntimeError("Qwen bypassed the generative evidence wrapper")
        hidden = output[0]
        ctx["core_length"] = hidden.shape[1]
        if ctx["outputs"] is not None:
            return
        if len(ctx["metadata"]) != hidden.shape[0]:
            raise ValueError("Metadata batch mismatch")
        outputs = []
        cuda_devices = [hidden.device.index or 0] if hidden.is_cuda else []
        # Evidence dropout does not perturb LoRA dropout or checkpoint RNG.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(ctx["seed"])
            for b, meta in enumerate(ctx["metadata"]):
                p = int(meta["prompt_length"])
                offset = int(meta.get("padding_offset", 0))
                groups = meta["word_token_indices"]
                if not groups:
                    raise ValueError("Empty original word sequence")
                words = []
                for indices in groups:
                    if not indices or min(indices) < offset or max(indices) >= offset + p:
                        raise ValueError("Word metadata reaches outside public prefix")
                    words.append(hidden[b, indices].float().mean(0))
                text = torch.stack(words)
                images, positions = [], []
                if len(meta["image_token_indices"]) != len(meta["image_grids"]):
                    raise ValueError("Image grid/token metadata mismatch")
                for indices, grid in zip(meta["image_token_indices"], meta["image_grids"]):
                    if not indices or min(indices) < offset or max(indices) >= offset + p:
                        raise ValueError("Image metadata reaches outside public prefix")
                    t, h, w = (int(v) for v in grid)
                    hm, wm = h // self.merge_size, w // self.merge_size
                    if h % self.merge_size or w % self.merge_size or len(indices) != t * hm * wm:
                        raise ValueError("Visual grid does not match native merged tokens")
                    y, x = torch.meshgrid(
                        torch.arange(hm, device=hidden.device),
                        torch.arange(wm, device=hidden.device),
                        indexing="ij",
                    )
                    xy = (
                        torch.stack(((x.float() + 0.5) / wm, (y.float() + 0.5) / hm), -1)
                        .reshape(-1, 2)
                        .repeat(t, 1)
                    )
                    images.append(hidden[b, indices].float())
                    positions.append(xy)
                if ctx["detach"]:
                    text = text.detach()
                    images = [v.detach() for v in images]
                result = self.evidence(text, images, positions, image_ids=meta.get("image_ids"))
                if result["memory"].ndim != 2 or not torch.isfinite(result["memory"]).all():
                    raise FloatingPointError("Invalid evidence memory")
                outputs.append(result)
        ctx["outputs"] = outputs

    def _enhance_head(self, _module, args):
        ctx = self._context
        if ctx is None or ctx["outputs"] is None:
            raise RuntimeError("Missing public-only evidence memory")
        states = args[0]
        if self._prewarm or self.run_config.get("disable_residual", False):
            return args
        if ctx["mode"] == "generate":
            # Native generation requests only the current next-token logits;
            # cached and uncached generation both select the final position.
            if states.shape[1] != 1:
                length = ctx["core_length"]
                pos = torch.arange(length - states.shape[1], length, device=states.device)
                active = torch.stack(
                    [
                        pos >= int(m["prompt_length"]) - 1 + int(m.get("padding_offset", 0))
                        for m in ctx["metadata"]
                    ]
                )
            else:
                active = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        else:
            keep = ctx["keep"]
            if isinstance(keep, torch.Tensor):
                pos = keep.to(states.device)
            elif isinstance(keep, int):
                length = ctx["core_length"]
                pos = torch.arange(length - states.shape[1], length, device=states.device)
            else:
                raise TypeError("Unsupported logits positions")
            active = torch.stack(
                [
                    pos >= int(m["prompt_length"]) - 1 + int(m.get("padding_offset", 0))
                    for m in ctx["metadata"]
                ]
            )
        enhanced = self.readout(
            states,
            [o["memory"] for o in ctx["outputs"]],
            active,
            [o.get("memory_bias") for o in ctx["outputs"]],
        )
        return (enhanced,) + args[1:]

    def forward(self, *args, public_metadata=None, detach_evidence_inputs=False, **kwargs):
        if self.evidence is None:
            result = self.generator(*args, **kwargs)
            result.evidence_outputs = []
            return result
        with self._request(
            public_metadata, "forward", kwargs.get("logits_to_keep", 0), detach_evidence_inputs
        ) as ctx:
            result = self.generator(*args, **kwargs)
            result.evidence_outputs = ctx["outputs"]
            return result

    def generate(self, *args, public_metadata=None, **kwargs):
        if kwargs.get("num_beams", 1) != 1:
            raise ValueError("Evidence request cache currently supports one beam")
        if self.training:
            raise RuntimeError("Call eval() before generation")
        kwargs.setdefault("logits_to_keep", 1)
        if self.evidence is None:
            return self.generator.generate(*args, **kwargs)
        with self._request(public_metadata, "generate", kwargs["logits_to_keep"]):
            return self.generator.generate(*args, **kwargs)

    def auxiliary_loss(self, outputs, gold_entities, metadata):
        if self.evidence is None:
            zero = next(self.generator.parameters()).new_zeros((), dtype=torch.float32)
            return {"entity": zero, "set": zero}
        if not (len(outputs) == len(gold_entities) == len(metadata)):
            raise ValueError("Auxiliary batch mismatch")
        values = [
            self.evidence.auxiliary_loss(o, gold, len(m["word_token_indices"]), len(m["image_ids"]))
            for o, gold, m in zip(outputs, gold_entities, metadata)
        ]
        return {key: torch.stack([v[key] for v in values]).mean() for key in ("entity", "set")}

    def save_pretrained(self, destination, **kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.generator.save_pretrained(destination, safe_serialization=True, **kwargs)
        payload = dict(
            schema="generative_eler_v1",
            variant=self.variant,
            rng_step=self._rng_step,
            evidence=self.evidence.state_dict() if self.evidence is not None else None,
            readout=self.readout.state_dict() if self.readout is not None else None,
        )
        torch.save(payload, destination / "evidence.pt")
        (destination / "run_config.json").write_text(
            json.dumps(self.run_config, ensure_ascii=False, indent=2) + "\n"
        )


def _attach_lora(base, config, peft):
    """Attach a fresh language adapter while leaving the image encoder frozen."""
    suffixes = tuple(
        "." + name
        for name in config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    )
    vision_parts = {"visual", "vision_model", "vision_tower"}
    targets = [
        name
        for name, _ in base.named_modules()
        if name.endswith(suffixes)
        and not (config.get("freeze_vision", True) and vision_parts.intersection(name.split(".")))
    ]
    if not targets:
        raise ValueError("No LoRA target modules matched the selected model")
    adapter = peft.LoraConfig(
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )
    return peft.get_peft_model(base, adapter)


def load_model(config, device, checkpoint=None):
    """Load a Qwen2.5-VL backbone, LoRA adapter, and optional ELER memory.

    model_id accepts a Hugging Face model name or a local model directory.
    checkpoint is a directory previously created by save_pretrained.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import peft

    cfg = dict(config)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        saved = json.loads((checkpoint / "run_config.json").read_text())
        cfg = {**saved, **cfg}
        architecture = (
            "variant",
            "hidden_dim",
            "entity_queries",
            "discovery_queries",
            "completion_queries",
            "attention_heads",
            "reader_layers",
            "feedforward_dim",
            "max_images",
        )
        for key in architecture:
            if key in config and key in saved and config[key] != saved[key]:
                raise ValueError(f"Checkpoint architecture mismatch: {key}")
    device = torch.device(device)
    dtype_name = cfg.get("compute_dtype", "bfloat16" if device.type == "cuda" else "float32")
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError("compute_dtype must be bfloat16, float16, or float32")
    dtype = dtypes[dtype_name]
    pretrained = {}
    if cfg.get("model_revision") is not None:
        pretrained["revision"] = cfg["model_revision"]
    if cfg.get("local_files_only", False):
        pretrained["local_files_only"] = True
    processor_source = (
        str(checkpoint)
        if checkpoint and (checkpoint / "preprocessor_config.json").exists()
        else cfg["model_id"]
    )
    processor = AutoProcessor.from_pretrained(
        processor_source,
        **pretrained,
        min_pixels=int(cfg.get("min_pixels", 56 * 56)),
        max_pixels=int(cfg.get("max_pixels", 448 * 448)),
    )
    base, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        **pretrained,
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        output_loading_info=True,
    )
    base.to(device)
    base.config.use_cache = False
    if checkpoint is None:
        generator = _attach_lora(base, cfg, peft)
    else:
        generator = peft.PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    if cfg.get("freeze_vision", True):
        for name, parameter in generator.named_parameters():
            if {"visual", "vision_model", "vision_tower"}.intersection(name.split(".")):
                parameter.requires_grad_(False)
    if cfg.get("gradient_checkpointing", True):
        generator.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        generator.enable_input_require_grads()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(cfg.get("seed", 1234)) + 8173)
        wrapper = GenerativeQwen(generator, cfg).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint / "evidence.pt", map_location="cpu", weights_only=True)
        if payload["schema"] != "generative_eler_v1" or payload["variant"] != wrapper.variant:
            raise ValueError("Evidence checkpoint architecture mismatch")
        if wrapper.evidence is not None:
            wrapper.evidence.load_state_dict(payload["evidence"], strict=True)
            wrapper.readout.load_state_dict(payload["readout"], strict=True)
        wrapper._rng_step = int(payload.get("rng_step", 0))
    metadata = dict(
        model_id=cfg["model_id"],
        variant=wrapper.variant,
        lora_parameters=sum(p.numel() for p in wrapper.lora_parameters()),
        module_parameters=sum(p.numel() for p in wrapper.module_parameters()),
        loading_info=json.loads(json.dumps(info, default=str)),
    )
    return processor, wrapper, metadata
