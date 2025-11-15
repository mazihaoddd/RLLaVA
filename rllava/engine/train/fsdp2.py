"""
Native FSDP training engine.

The class mirrors the `TrainEngine` interface and prepares FSDP-specific
configuration that will be consumed when models/optimizers are attached.
Subsequent steps will extend this skeleton with full training utilities.
"""

from __future__ import annotations

import os
import logging
import torch
import torch.distributed as dist
import torch.nn as nn

from abc import ABC
from contextlib import contextmanager, nullcontext
from packaging import version
from typing import Any
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.device import get_device_id, get_torch_device
from rllava.engine.train.fsdp import FSDPAccelerator, enable_activation_offloading
from .. import register_engine

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp.fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


@register_engine("fsdp2")
class FSDP2Accelerator(FSDPAccelerator):
    """Native FSDP engine used by PPO-style training loops.

    The engine mirrors the ``TrainEngine`` interface and exposes a compact API
    that the actor/critic roles rely on.  Highlights:

    * Transparent wrapping of modules with :class:`FullyShardedDataParallel`.
    * Helper utilities to gather parameters for generation/export and to clip
      gradients across the participating FSDP modules.
    * Lightweight checkpoint orchestration that persists model, optimizer and
      scheduler states without depending on ``accelerate``.

    The implementation is intentionally minimalist so that more advanced
    sharding behaviour borrowed from external projects can be integrated later
    without breaking the current training stack.
    """

    def get_fsdp_state_ctx(self, model, state_type, state_cfg, optim_cfg):
        return nullcontext()

    def clip_grad_norm_(self, model: FSDPModule, max_norm: float):
        grad_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=max_norm)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        
        return grad_norm

    @contextmanager
    def eval(self, model: FSDP):
        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        model.eval()

        yield

        if self.world_size > 1:
            model.reshard()

        if self.fsdp_config.offload_params:
           self.offload_fsdp_model_to_cpu(model)

    @contextmanager
    def train(self, model: FSDP, optimizer: Optimizer):
        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        if self.fsdp_config.offload_optimizer:
            self.load_fsdp_optimizer(optimizer, device_id=get_device_id())
        model.train()

        yield

        if self.world_size > 1:
            model.reshard()

        if self.fsdp_config.offload_params:
            self.offload_fsdp_model_to_cpu(model)
        if self.fsdp_config.offload_optimizer:
            self.offload_fsdp_optimizer(optimizer)

    def _prepare_module(self, module: nn.Module, **kwargs: Any):
        from rllava.utils.torch_dtypes import PrecisionType

        self.wait_for_everyone()

        forward_only = kwargs.get("forward_only", False)

        mixed_precision_config = self.fsdp_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32

        cpu_offload = None
        if forward_only or self.fsdp_config.offload_params:
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            self.fsdp_config.offload_params = False
            self.fsdp_config.offload_optimizer = False

        assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
        )

        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": self.fsdp_config.reshard_after_forward,
            "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
        }
        full_state = module.state_dict()
        apply_fsdp2(module, fsdp_kwargs, self.fsdp_config)
        fsdp2_load_full_state_dict(module, full_state, self.device_mesh, cpu_offload)

        if module not in self.wrapped_models:
            self.wrapped_models.append(module)

        if self.config.model.enable_activation_offload:
            enable_gradient_checkpointing = self.config.model.enable_gradient_checkpointing
            enable_activation_offloading(module, 'fsdp', enable_gradient_checkpointing)

        if not forward_only:
            if self.fsdp_config.offload_params:
                self.offload_fsdp_model_to_cpu(module)
                log_gpu_memory_usage("After offload model during init", logger=logger)

        return module

    @torch.no_grad()
    def load_fsdp_model_to_gpu(self, model: FSDP):
        device = get_device_id()
        model.to(device)
    
    @torch.no_grad()
    def offload_fsdp_model_to_cpu(self, model: FSDP, empty_cache: bool = True):
        model.cpu()
        if empty_cache:
            get_torch_device().empty_cache()


def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""

    def shard_placement_fn(param):
        shape = list(param.shape)
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return shard_placement_fn

@contextmanager
def maybe_patch_fsdp_module(model):
    if fully_shard_module is None:
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module

def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.wrap_policy.get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module

def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """

    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:
        # official torch 2.6.0 set_model_state_dict API leads to OOM
        # use torch 2.7.0 copy from verl/third_party/torch/distributed/checkpoint
        from rllava.engine.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=get_device_id(), non_blocking=True)
    else:
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())

def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm