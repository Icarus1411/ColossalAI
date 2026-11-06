import torch

from colossalai.interface import OptimizerWrapper

from .fp16_torch import TorchAMPModule
from .mixed_precision_base import MixedPrecision


class BF16MixedPrecision(MixedPrecision):
    """BF16 autocast with FP32 parameters and unscaled gradients."""

    def configure(self, model, optimizer=None, criterion=None):
        model = TorchAMPModule(model, dtype=torch.bfloat16)
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer = OptimizerWrapper(optimizer)
        if criterion is not None:
            criterion = TorchAMPModule(criterion, dtype=torch.bfloat16)
        return model, optimizer, criterion
