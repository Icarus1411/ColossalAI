"""Qwen3.5 dense, full-sequence text training with Gated DeltaNet at TP=1."""

import inspect
from operator import attrgetter
from functools import wraps

from colossalai.shardformer.layer import FusedGatedDeltaNet

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


def text_training_forward(native_forward):
    signature = inspect.signature(native_forward)

    @wraps(native_forward)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.update(arguments.pop("kwargs", {}))
        if arguments.get("use_cache") or arguments.get("past_key_values") is not None:
            raise NotImplementedError("Qwen3.5 policy supports full-sequence training only; use_cache=False is required")
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            if arguments.get(key) is not None:
                raise NotImplementedError("Qwen3.5 policy currently supports text-only inputs")
        for key in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens"):
            if arguments.get(key) is not None:
                raise NotImplementedError("Qwen3.5 policy does not support packed sequences")
        return native_forward(self, *args, **kwargs)

    return forward


class Qwen3_5Policy(Policy):
    """Replace only GDN; retain HF attention, MLP, parameter names and tied weights.

    ConditionalGeneration keeps its vision tower and checkpoint layout, freezes
    the tower, and rejects visual inputs. Construct optimizers after optimizing.
    """

    def config_sanity_check(self):
        from packaging.version import Version
        from transformers import __version__

        if Version(__version__) < Version("5.17.0"):
            raise NotImplementedError("Qwen3.5 policy requires transformers>=5.17.0")
        config = self.shard_config
        if config.tensor_parallel_size != 1 or config.pipeline_stage_manager is not None:
            raise NotImplementedError("Qwen3.5 currently supports TP=1 and no pipeline parallelism")
        if config.enable_sequence_parallelism or config.expert_parallel_size != 1:
            raise NotImplementedError("Qwen3.5 currently supports data parallelism only")
        for key in ("enable_fused_normalization", "enable_flash_attention", "enable_jit_fused", "fp8_communication"):
            if getattr(config, key):
                raise NotImplementedError(f"Qwen3.5 policy has not implemented {key}; leave it disabled")

    def preprocess(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        self.model.config.use_cache = False
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = False
        for module in self.model.modules():
            if isinstance(module, Qwen3_5VisionModel):
                module.requires_grad_(False)
                module._colossalai_fsdp_frozen_tower = True
        return self.model

    def module_policy(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5Model,
            Qwen3_5TextModel,
        )

        policy = {
            Qwen3_5DecoderLayer: ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription("linear_attn", FusedGatedDeltaNet,
                                                    kwargs={"force_fp32_core": True}, ignore_if_not_exist=True,
                                                    can_reuse=attrgetter("force_fp32_core"))
                ]
            )
        }
        for cls in (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5Model, Qwen3_5TextModel):
            policy[cls] = ModulePolicyDescription(method_replacement={"forward": text_training_forward(cls.forward)})
        return policy

    def postprocess(self):
        return self.model


class Qwen3_5ForCausalLMPolicy(Qwen3_5Policy):
    pass


class Qwen3_5ForConditionalGenerationPolicy(Qwen3_5Policy):
    pass
