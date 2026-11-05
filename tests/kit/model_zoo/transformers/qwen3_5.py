"""Offline tiny Qwen3.5 dense entries for text-only L1/L2 (not full-size checkpoints)."""

from functools import partial

import torch
import torch.nn.functional as F
import transformers

from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.spec import TokenLoss

from ..registry import ModelAttribute, model_zoo


def text_config(value_heads=2, hidden_size=64):
    return transformers.Qwen3_5TextConfig(
        vocab_size=256, hidden_size=hidden_size, intermediate_size=hidden_size * 2, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=value_heads,
        linear_key_head_dim=64, linear_value_head_dim=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=256, use_cache=False, pad_token_id=0, tie_word_embeddings=True,
        attn_implementation="eager", attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [6, 5, 5]},
    )


def prepare_text(model):
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)
    return model


def build_model(value_heads=2, conditional=False, fused=False, hidden_size=64):
    from .qwen3_5_reference import native_reference

    config = text_config(value_heads, hidden_size)
    if conditional:
        config = transformers.Qwen3_5Config(
            text_config=config.to_dict(),
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": hidden_size, "num_position_embeddings": 16},
            image_token_id=253, video_token_id=254, vision_start_token_id=252,
            vision_end_token_id=255, tie_word_embeddings=True,
        )
        model = transformers.Qwen3_5ForConditionalGeneration(config)
    else:
        model = transformers.Qwen3_5ForCausalLM(config)
    model = prepare_text(native_reference(model))
    if fused:
        from colossalai.shardformer import ShardConfig, ShardFormer

        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    return model


def data_gen():
    ids = torch.randint(1, 250, (2, 65))
    mask = torch.ones_like(ids)
    # Variable valid lengths exercise token weighting across micro-batches.
    left, right = torch.randint(1, 9, (2,)).tolist()
    mask[0, :left] = 0
    mask[1, -right:] = 0
    ids = ids.masked_fill(mask == 0, 0)
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def outputs(output):
    return {"logits": output.logits, "loss": output.loss}


def loss_fn(output):
    return output["loss"]


def token_loss(output, batch):
    logits = output.logits[:, :-1].flatten(0, 1)
    return F.cross_entropy(logits.float(), batch["labels"][:, 1:].flatten(), ignore_index=-100, reduction="sum")


def token_count(batch):
    return (batch["labels"][:, 1:] != -100).sum()


if hasattr(transformers, "Qwen3_5ForCausalLM"):
    # Both real 0.8B and 2B checkpoints use equal key/value head counts.
    # Tiny entries retain that ratio, with widths scaled 1:2 as in the checkpoints.
    for label, width in (("0_8b", 64), ("2b", 128)):
        for conditional in (False, True):
            name = f"transformers_qwen3_5_{label}_tiny" + ("_conditional" if conditional else "")
            reference = partial(build_model, hidden_size=width, conditional=conditional)
            for fused in (False, True):
                spec = NumericsSpec(
                    reference_fn=reference if fused else None,
                    prepare_fn=prepare_text,
                    token_loss=TokenLoss(token_loss, token_count),
                )
                model_zoo.register(
                    name=name + ("_fused" if fused else ""),
                    model_fn=partial(build_model, hidden_size=width, conditional=conditional, fused=fused),
                    data_gen_fn=data_gen, output_transform_fn=outputs, loss_fn=loss_fn,
                    model_attribute=ModelAttribute(has_control_flow=True, numerics=spec),
                )
