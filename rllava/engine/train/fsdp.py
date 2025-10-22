"""
Native FSDP training engine.

The class mirrors the `TrainEngine` interface and prepares FSDP-specific
configuration that will be consumed when models/optimizers are attached.
Subsequent steps will extend this skeleton with full training utilities.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
try:  # torch >= 2.4
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # torch < 2.4
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore

from rllava.engine.train.base import TrainEngine
from .. import register_engine


try:
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FlatParameter
except Exception:  # pragma: no cover - handled dynamically at runtime
    CPUOffload = None  # type: ignore
    MixedPrecision = None  # type: ignore
    FullyShardedDataParallel = None  # type: ignore
    ShardingStrategy = None  # type: ignore
    FlatParameter = nn.Parameter  # type: ignore


def _resolve_torch_dtype(dtype_str: Optional[str], fallback: torch.dtype = torch.float32) -> torch.dtype:
    if not dtype_str:
        return fallback
    normalized = dtype_str.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "tf32": torch.float32,
    }
    return mapping.get(normalized, fallback)


class _FSDPOffloadManager:
    """Lightweight CPU offload helper for FSDP modules and optimizers."""

    def __init__(self, module: nn.Module, optimizers: list[Optimizer], grad_offload: bool, optim_offload: bool):
        self._module = module
        self._optimizers = optimizers
        self._grad_offload = grad_offload
        self._optim_offload = optim_offload
        self._cached_param_devices: dict[nn.Parameter, torch.device] = {}
        self._cached_grad_devices: dict[nn.Parameter, torch.device] = {}

    def _iter_params(self):
        params = list(self._module.parameters())
        flat_params = [p for p in params if isinstance(p, FlatParameter)]
        if flat_params:
            return flat_params
        return params

    def offload_parameters(self) -> None:
        for param in self._iter_params():
            if not param.is_cuda:
                continue
            self._cached_param_devices[param] = param.device
            param.data = param.data.cpu()

    def load_parameters(self, non_blocking: bool = True) -> None:
        for param in self._iter_params():
            last_device = self._cached_param_devices.get(param, None)
            if last_device is None or param.is_cuda:
                continue
            param.data = param.data.to(last_device, non_blocking=non_blocking)

    def offload_gradients(self) -> None:
        if not self._grad_offload:
            return
        for param in self._iter_params():
            grad = param.grad
            if grad is None or not grad.is_cuda:
                continue
            self._cached_grad_devices[param] = grad.device
            param.grad = grad.cpu()

    def load_gradients(self, non_blocking: bool = True) -> None:
        if not self._grad_offload:
            return
        for param in self._iter_params():
            grad = param.grad
            if grad is None or grad.is_cuda:
                continue
            target_device = self._cached_grad_devices.get(param, None)
            if target_device is None:
                target_device = self._cached_param_devices.get(param, None)
            if target_device is None:
                continue
            param.grad = grad.to(target_device, non_blocking=non_blocking)

    def offload_optimizers(self) -> None:
        if not self._optim_offload:
            return
        for optimizer in self._optimizers:
            self._move_optimizer_state(optimizer, torch.device("cpu"))

    def load_optimizers(self, non_blocking: bool = True) -> None:
        if not self._optim_offload:
            return
        target = self._infer_device()
        if target is None:
            return
        for optimizer in self._optimizers:
            self._move_optimizer_state(optimizer, target, non_blocking=non_blocking)

    def _infer_device(self) -> Optional[torch.device]:
        for param in self._iter_params():
            if param.is_cuda:
                return param.device
            if param in self._cached_param_devices:
                return self._cached_param_devices[param]
        return None

    def _move_optimizer_state(self, optimizer: Optimizer, device: torch.device, non_blocking: bool = True) -> None:
        state = optimizer.state
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=non_blocking)
            elif isinstance(value, dict):
                for inner_key, inner_val in value.items():
                    if isinstance(inner_val, torch.Tensor):
                        value[inner_key] = inner_val.to(device, non_blocking=non_blocking)


@register_engine("fsdp")
class FSDPAccelerator(TrainEngine):
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

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "strategy", None) not in {"fsdp", "fsdp2", None}:
            raise ValueError(f"Unsupported training strategy: {config.strategy}")

        self.fsdp_config = getattr(config, "fsdp", None)
        if self.fsdp_config is None:
            raise ValueError("FSDP engine requires `config.fsdp` to be provided.")

        self._mixed_precision_config = self._build_mixed_precision_config()
        self._cpu_offload_config = self._build_cpu_offload_config()
        self._use_orig_params = getattr(self.fsdp_config, "use_orig_params", False)
        self._wrapped_models = []
        self._optimizers: list[Optimizer] = []
        self._lr_schedulers: list[LRScheduler] = []
        self._state_filename = "fsdp_engine_state.pt"
        self._offload_param = getattr(self.fsdp_config, "offload_params", False)
        self._offload_optimizer = getattr(self.fsdp_config, "offload_optimizer", False)
        self._offload_gradients = self._offload_param and getattr(self.fsdp_config, "enable_cpu_offload", False)
        self._offload_managers: list[_FSDPOffloadManager] = []
        
        # LoRA configuration
        model_config = getattr(config, "model", None)
        self._lora_rank = getattr(model_config, "lora_rank", 0) if model_config else 0
        self._is_lora = self._lora_rank > 0
        self._lora_config = None
        if self._is_lora and model_config:
            self._lora_config = {
                "lora_rank": getattr(model_config, "lora_rank", 8),
                "lora_alpha": getattr(model_config, "lora_alpha", 16),
                "target_modules": getattr(model_config, "target_modules", None),
                "exclude_modules": getattr(model_config, "exclude_modules", None),
            }

    # -------------------------------------------------------------------------
    # Distributed metadata helpers
    # -------------------------------------------------------------------------
    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def num_processes(self) -> int:
        return self.world_size

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    # -------------------------------------------------------------------------
    # Synchronization helpers
    # -------------------------------------------------------------------------
    def wait_for_everyone(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------
    def get_init_weight_context(self, use_meta_tensor: bool = True):
        """
        Provide an initialization context manager that mirrors verl's implementation.

        This method mirrors the behavior of verl's get_init_weight_context_manager.
        When use_meta_tensor=False (e.g., when model has tie_word_embeddings=True),
        we avoid meta tensor initialization to preserve weight tying.
        
        When use_meta_tensor=True, non-rank-0 processes use meta tensor to save memory,
        and rank-0 loads actual weights which are then synced by FSDP.
        
        Args:
            use_meta_tensor: Whether to use meta tensor initialization for non-rank-0 processes.
                           Set to False when model has weight tying (tie_word_embeddings).
        
        Returns:
            A context manager or callable that can be invoked with ()
        """
        from accelerate import init_empty_weights
        
        cpu_init_weights = lambda: torch.device("cpu")
        if use_meta_tensor:
            init_context = init_empty_weights if (torch.distributed.is_initialized() and self.rank != 0) else cpu_init_weights
        else:
            init_context = cpu_init_weights
        return init_context

    # -------------------------------------------------------------------------
    # Placeholder interfaces to be filled in subsequent steps
    # -------------------------------------------------------------------------
    def prepare(self, *args: Any, **kwargs: Any):
        """Wrap supported training artefacts with FSDP.

        Parameters are processed in-order to mirror the ``accelerate`` API.
        Supported artefacts:

        - ``nn.Module`` instances are wrapped with FSDP according to the
          configuration provided via ``config.fsdp``.
        - ``Optimizer`` / ``LRScheduler`` instances are cached so their state
          can be saved/restored alongside the model.
        """
        if kwargs:
            raise NotImplementedError("Keyword argument handling for prepare is not implemented yet.")

        prepared_items = [self._prepare_item(arg) for arg in args]
        if len(prepared_items) == 1:
            return prepared_items[0]
        return tuple(prepared_items)

    def unwrap_model(self, model):
        if isinstance(model, FullyShardedDataParallel):
            return model.module
        return model

    def unwrap_model_for_generation(self, model, is_peft_model: bool = False):
        base_model = self.unwrap_model(model)

        @contextmanager
        def _ctx():
            adapter_ctx = nullcontext()
            if is_peft_model and hasattr(base_model, "disable_adapter"):
                adapter_ctx = base_model.disable_adapter()

            with adapter_ctx:
                # For FSDP, directly yield the wrapped model
                # FSDP will handle state_dict() internally without deadlocks
                # This matches the behavior of HFAccelerator
                yield model

        return _ctx

    def load_state(self, checkpoint_path: str):
        """Restore model/optimizer/scheduler states from ``checkpoint_path``."""
        state_file = os.path.join(checkpoint_path, self._state_filename)
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"Checkpoint file not found: {state_file}")

        payload = torch.load(state_file, map_location="cpu")

        model_states = payload.get("models", [])
        if len(model_states) != len(self._wrapped_models):
            raise RuntimeError(
                "Mismatch between saved model count and currently wrapped models:"
                f" {len(model_states)} vs {len(self._wrapped_models)}"
            )

        for module, module_state in zip(self._wrapped_models, model_states):
            self._load_model_state(module, module_state)

        optimizer_states = payload.get("optimizers", [])
        if len(optimizer_states) != len(self._optimizers):
            raise RuntimeError(
                "Mismatch between saved optimizer count and registered optimizers:"
                f" {len(optimizer_states)} vs {len(self._optimizers)}"
            )
        for optimizer, optimizer_state in zip(self._optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)

        scheduler_states = payload.get("lr_schedulers", [])
        if len(scheduler_states) != len(self._lr_schedulers):
            raise RuntimeError(
                "Mismatch between saved LR scheduler count and registered schedulers:"
                f" {len(scheduler_states)} vs {len(self._lr_schedulers)}"
            )
        for scheduler, scheduler_state in zip(self._lr_schedulers, scheduler_states):
            scheduler.load_state_dict(scheduler_state)

        self.wait_for_everyone()
        
        # Load LoRA adapter if enabled
        if self._is_lora:
            self._load_lora_adapter(checkpoint_path)

    def save_state(self, checkpoint_path: str):
        """Persist model/optimizer/scheduler states to ``checkpoint_path``."""
        if not self._wrapped_models:
            raise RuntimeError("No models have been prepared with FSDP. Call prepare(model) before save_state().")

        os.makedirs(checkpoint_path, exist_ok=True)

        payload = {
            "models": [self._gather_model_state(module) for module in self._wrapped_models],
            "optimizers": [optimizer.state_dict() for optimizer in self._optimizers],
            "lr_schedulers": [scheduler.state_dict() for scheduler in self._lr_schedulers],
        }

        if self.is_main_process:
            state_file = os.path.join(checkpoint_path, self._state_filename)
            torch.save(payload, state_file)

        self.wait_for_everyone()
        
        # Save LoRA adapter if enabled
        if self._is_lora:
            self._save_lora_adapter(checkpoint_path)

    def backward(self, loss: torch.Tensor):
        # FSDP handles parameter offload automatically
        # We only manage optimizer offload if configured
        if self._offload_optimizer:
            for manager in self._offload_managers:
                manager.load_optimizers()
        
        loss.backward()
        
        if self._offload_optimizer:
            for manager in self._offload_managers:
                manager.offload_optimizers()

    def clip_grad_norm_(self, parameters, max_norm: float):
        # FSDP handles parameter/gradient access automatically
        # No need to manually load/offload
        if parameters is None:
            if not self._wrapped_models:
                result = torch.tensor(0.0, device=self._default_device())
            else:
                norms = []
                for module in self._wrapped_models:
                    norm = module.clip_grad_norm_(max_norm)
                    norms.append(self._ensure_tensor(norm))
                result = torch.stack(norms).max()
        else:
            if isinstance(parameters, nn.Module):
                parameters = parameters.parameters()

            if isinstance(parameters, (list, tuple)):
                iterable: Iterable[torch.Tensor] = parameters
            else:
                iterable = list(parameters)

            result = torch.nn.utils.clip_grad_norm_(iterable, max_norm)
            result = self._ensure_tensor(result)
        
        return result

    def get_model_weights(self, model=None):
        """Iterate over ``(name, tensor)`` pairs of CPU tensors.

        The iterator is compatible with the streaming logic used by inference
        engines—downstream consumers can iterate once and ship the tensors
        without loading the entire state dict into memory.
        """
        target_model = model or (self._wrapped_models[0] if self._wrapped_models else None)
        if target_model is None:
            raise RuntimeError("No model available to extract weights from. Call prepare(model) first or pass a model.")

        def _state_dict():
            # Simply gather the model state and yield
            # FSDP will handle parameter gathering internally
            state_dict = self._gather_model_state(target_model)
            
            for name, tensor in state_dict.items():
                yield name, tensor

        return _state_dict()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _build_mixed_precision_config(self) -> Optional[Dict[str, torch.dtype]]:
        mp_param = getattr(self.fsdp_config, "mp_param_dtype", None)
        mp_reduce = getattr(self.fsdp_config, "mp_reduce_dtype", None)
        mp_buffer = getattr(self.fsdp_config, "mp_buffer_dtype", None)
        if not any([mp_param, mp_reduce, mp_buffer]):
            return None
        return {
            "param_dtype": _resolve_torch_dtype(mp_param, torch.bfloat16),
            "reduce_dtype": _resolve_torch_dtype(mp_reduce, torch.float32),
            "buffer_dtype": _resolve_torch_dtype(mp_buffer, torch.float32),
        }

    def _build_cpu_offload_config(self) -> Optional[Any]:
        enable_cpu_offload = getattr(self.fsdp_config, "enable_cpu_offload", False)
        if not enable_cpu_offload or CPUOffload is None:
            return None
        return CPUOffload(offload_params=True)

    def _prepare_item(self, obj: Any):
        if isinstance(obj, nn.Module):
            return self._prepare_module(obj)
        if isinstance(obj, Optimizer):
            self._register_optimizer(obj)
            return obj
        if isinstance(obj, LRScheduler):
            self._register_lr_scheduler(obj)
            return obj
        return obj

    def _prepare_module(self, module: nn.Module):
        if FullyShardedDataParallel is None:
            raise RuntimeError("torch.distributed.fsdp is not available in this environment.")

        # Apply LoRA if configured and not already a PeftModel
        if self._is_lora:
            # Check if module is already a PeftModel (to avoid double application)
            is_peft_model = hasattr(module, "peft_config") and hasattr(module, "base_model")
            if not is_peft_model:
                module = self._apply_lora(module)
            elif self.is_main_process:
                print("Model is already a PeftModel, skipping LoRA application in FSDP")

        fsdp_kwargs: Dict[str, Any] = {
            "use_orig_params": self._use_orig_params,
        }
        
        # Add param_init_fn to handle meta tensor materialization
        # This is critical for reducing peak memory usage when using meta tensor initialization
        def param_init_fn(sub_module: nn.Module) -> nn.Module:
            """
            Initialize parameters for FSDP, converting meta tensors to empty tensors on GPU.
            
            This function is called by FSDP during initialization to materialize meta tensors
            on non-rank-0 processes. By using to_empty(), we avoid loading full weights to GPU
            before sharding, significantly reducing peak memory usage.
            
            Mimics verl's init_fn from fsdp_utils.py
            """
            if dist.is_initialized() and dist.get_rank() != 0:
                device_id = self._infer_device_id()
                if device_id is not None:
                    device = torch.device("cuda", device_id)
                    sub_module = sub_module.to_empty(device=device, recurse=False)
                    torch.cuda.empty_cache()
            return sub_module
        
        fsdp_kwargs["param_init_fn"] = param_init_fn
        
        # Enable sync_module_states to broadcast weights from rank 0 to other ranks
        # This works together with param_init_fn and meta tensor initialization:
        # - rank 0: loads full weights on CPU
        # - rank != 0: has meta tensors materialized to empty tensors by param_init_fn
        # - sync_module_states=True: FSDP broadcasts rank 0's weights to all other ranks
        fsdp_kwargs["sync_module_states"] = True

        # Add auto_wrap_policy to handle large models layer-by-layer
        # This is CRITICAL for avoiding OOM on rank 0 during FSDP initialization
        # Without this, FSDP tries to process the entire model at once, causing peak memory issues
        auto_wrap_policy = self._get_auto_wrap_policy(module)
        if auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy

        if self._mixed_precision_config and MixedPrecision is not None:
            fsdp_kwargs["mixed_precision"] = MixedPrecision(**self._mixed_precision_config)

        if self._cpu_offload_config is not None:
            fsdp_kwargs["cpu_offload"] = self._cpu_offload_config

        if getattr(self.fsdp_config, "enable_full_shard", True) is False and ShardingStrategy is not None:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy.SHARD_GRAD_OP
        else:
            # Explicitly set FULL_SHARD as default sharding strategy
            if ShardingStrategy is not None:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD

        device_id = self._infer_device_id()
        if device_id is not None:
            fsdp_kwargs["device_id"] = device_id

        wrapped_module = FullyShardedDataParallel(module, **fsdp_kwargs)
        self._wrapped_models.append(wrapped_module)
        
        # Register offload manager for optimizer offload only
        # FSDP handles parameter offload internally via cpu_offload config
        if self._offload_optimizer:
            self._register_offload_manager(wrapped_module)
        
        return wrapped_module
    
    def _get_auto_wrap_policy(self, module: nn.Module):
        """
        Get auto wrap policy for FSDP to handle large models layer-by-layer.
        
        This prevents OOM by wrapping each transformer layer separately, so FSDP
        can process and shard them one at a time instead of loading the entire model.
        
        Mimics verl's get_fsdp_wrap_policy implementation.
        """
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
        from functools import partial
        from transformers.trainer_pt_utils import get_module_class_from_name
        
        # First, try to use the model's _no_split_modules attribute
        # This is the standard HuggingFace way to define which layers should not be split
        default_transformer_cls_names = getattr(module, "_no_split_modules", None)
        
        if default_transformer_cls_names:
            # Use the model's own definition of transformer layers
            transformer_cls_to_wrap = set()
            for layer_class_name in default_transformer_cls_names:
                transformer_cls = get_module_class_from_name(module, layer_class_name)
                if transformer_cls is not None:
                    transformer_cls_to_wrap.add(transformer_cls)
            
            if transformer_cls_to_wrap:
                if self.rank == 0:
                    print(f"[FSDP] Using transformer_auto_wrap_policy with layers from _no_split_modules: {transformer_cls_to_wrap}")
                return partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
        
        # Fallback: manually detect common transformer layer classes
        transformer_layer_cls = set()
        layer_class_names = [
            "Qwen2VLDecoderLayer",
            "Qwen2DecoderLayer",
            "LlamaDecoderLayer",
            "MistralDecoderLayer",
            "GPT2Block",
            "BertLayer",
            "T5Block",
        ]
        
        for sub_module in module.modules():
            module_class_name = sub_module.__class__.__name__
            if module_class_name in layer_class_names:
                transformer_layer_cls.add(sub_module.__class__)
        
        if transformer_layer_cls:
            if self.rank == 0:
                print(f"[FSDP] Using transformer_auto_wrap_policy with manually detected layers: {transformer_layer_cls}")
            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            )
        
        # Final fallback: use size-based wrapping
        # Wrap modules with >= 100M parameters
        if self.rank == 0:
            print("[FSDP] Using size_based_auto_wrap_policy with min_num_params=100M")
        
        return partial(size_based_auto_wrap_policy, min_num_params=100_000_000)

    def _infer_device_id(self) -> Optional[int]:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return None

    def _default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self._default_device())
        return torch.tensor(value, device=self._default_device())

    def _register_optimizer(self, optimizer: Optimizer) -> None:
        if optimizer not in self._optimizers:
            self._optimizers.append(optimizer)
            # Offload optimizer if configured and manager exists
            if self._offload_optimizer:
                for manager in self._offload_managers:
                    manager.offload_optimizers()

    def _register_lr_scheduler(self, scheduler: LRScheduler) -> None:
        if scheduler not in self._lr_schedulers:
            self._lr_schedulers.append(scheduler)

    def _register_offload_manager(self, module: nn.Module) -> None:
        # Only create manager for optimizer offload
        # Parameter offload is handled by FSDP's built-in cpu_offload
        manager = _FSDPOffloadManager(
            module,
            self._optimizers,
            grad_offload=False,  # Disable grad offload, let FSDP handle it
            optim_offload=self._offload_optimizer,
        )
        self._offload_managers.append(manager)
        # Don't offload parameters here - FSDP will handle it
        # Only offload optimizer if configured
        if self._offload_optimizer:
            manager.offload_optimizers()

    def _gather_model_state(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """Gather a full state dict on CPU regardless of local sharding.
        
        This method must be called synchronously on all ranks to avoid deadlocks.
        """
        if isinstance(module, FullyShardedDataParallel):
            # Use FSDP's recommended state_dict API
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            
            # rank0_only=False ensures all ranks get the full state dict
            # This is needed when multiple ranks may call state_dict() independently (e.g., vLLM)
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = module.state_dict()
        else:
            state_dict = module.state_dict()

        return {key: tensor.detach().cpu() for key, tensor in state_dict.items()}

    def _load_model_state(self, module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a CPU state dict into a (possibly sharded) module."""
        if isinstance(module, FullyShardedDataParallel):
            with FullyShardedDataParallel.summon_full_params(module, recurse=True):
                module.load_state_dict(state_dict)
        else:
            module.load_state_dict(state_dict)
        # FSDP will handle parameter offload automatically after loading
    
    # -------------------------------------------------------------------------
    # LoRA utilities
    # -------------------------------------------------------------------------
    def _apply_lora(self, module: nn.Module) -> nn.Module:
        """Apply LoRA to the module.
        
        Args:
            module: The module to apply LoRA to
            
        Returns:
            The module with LoRA applied (PeftModel)
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("peft library is required for LoRA. Install it with: pip install peft")
        
        if self._lora_config is None:
            raise ValueError("LoRA is enabled but lora_config is not set")
        
        if self.is_main_process:
            print("Applying LoRA to module")
        
        # Enable gradient computation for input embeddings if needed
        module.enable_input_require_grads()
        
        # Prepare LoRA configuration
        lora_config_dict = {
            "task_type": TaskType.CAUSAL_LM,
            "r": self._lora_config["lora_rank"],
            "lora_alpha": self._lora_config["lora_alpha"],
            "bias": "none",
        }
        
        if self._lora_config["target_modules"] is not None:
            lora_config_dict["target_modules"] = self._lora_config["target_modules"]
        
        if self._lora_config["exclude_modules"] is not None:
            lora_config_dict["exclude_modules"] = self._lora_config["exclude_modules"]
        
        lora_config = LoraConfig(**lora_config_dict)
        module = get_peft_model(module, lora_config)
        
        if self.is_main_process:
            print(f"LoRA applied: r={self._lora_config['lora_rank']}, alpha={self._lora_config['lora_alpha']}")
            module.print_trainable_parameters()
        
        return module
    
    def get_lora_weights(self, model=None) -> Optional[Dict[str, torch.Tensor]]:
        """Get LoRA adapter weights from the model.
        
        Args:
            model: The model to extract LoRA weights from. If None, uses the first wrapped model.
            
        Returns:
            Dictionary of LoRA weights, or None if LoRA is not enabled
        """
        if not self._is_lora:
            return None
        
        target_model = model or (self._wrapped_models[0] if self._wrapped_models else None)
        if target_model is None:
            raise RuntimeError("No model available to extract LoRA weights from")
        
        # Check if the model is a PeftModel
        unwrapped = self.unwrap_model(target_model)
        if not hasattr(unwrapped, "peft_config"):
            return None
        
        try:
            from peft.utils.save_and_load import get_peft_model_state_dict
            
            if isinstance(target_model, FullyShardedDataParallel):
                # Use FSDP's state_dict API to gather LoRA parameters
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FullyShardedDataParallel.state_dict_type(
                    target_model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    full_state_dict = target_model.state_dict()
                    # Extract only LoRA parameters
                    lora_state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict)
            else:
                lora_state_dict = get_peft_model_state_dict(unwrapped)
            
            # Ensure all tensors are on CPU
            lora_state_dict = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in lora_state_dict.items()
            }
            
            return lora_state_dict
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Failed to extract LoRA weights: {e}")
            return None
    
    def _save_lora_adapter(self, checkpoint_path: str):
        """Save LoRA adapter weights and configuration.
        
        Args:
            checkpoint_path: Directory to save the LoRA adapter
        """
        if not self._wrapped_models:
            return
        
        lora_save_path = os.path.join(checkpoint_path, "lora_adapter")
        
        # Get the first wrapped model
        target_model = self._wrapped_models[0]
        unwrapped = self.unwrap_model(target_model)
        
        # Check if model has peft_config
        if not hasattr(unwrapped, "peft_config"):
            if self.is_main_process:
                print("Warning: Model does not have peft_config, skipping LoRA adapter save")
            return
        
        try:
            # Save configuration on rank 0
            peft_config_dict = {}
            if self.is_main_process:
                os.makedirs(lora_save_path, exist_ok=True)
                
                from dataclasses import asdict
                peft_config = unwrapped.peft_config.get("default")
                if peft_config:
                    peft_config_dict = asdict(peft_config)
                    # Convert enum values to strings
                    if "task_type" in peft_config_dict and hasattr(peft_config_dict["task_type"], "value"):
                        peft_config_dict["task_type"] = peft_config_dict["task_type"].value
                    if "peft_type" in peft_config_dict and hasattr(peft_config_dict["peft_type"], "value"):
                        peft_config_dict["peft_type"] = peft_config_dict["peft_type"].value
                    if "target_modules" in peft_config_dict and isinstance(peft_config_dict["target_modules"], set):
                        peft_config_dict["target_modules"] = list(peft_config_dict["target_modules"])
            
            # Get LoRA weights
            lora_weights = self.get_lora_weights(target_model)
            
            if lora_weights and self.is_main_process:
                # Save weights using safetensors
                try:
                    from safetensors.torch import save_file
                    save_file(lora_weights, os.path.join(lora_save_path, "adapter_model.safetensors"))
                except ImportError:
                    # Fallback to torch.save
                    torch.save(lora_weights, os.path.join(lora_save_path, "adapter_model.bin"))
                
                # Save config
                import json
                with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                    json.dump(peft_config_dict, f, ensure_ascii=False, indent=4)
                
                print(f"LoRA adapter saved to: {lora_save_path}")
        
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Failed to save LoRA adapter: {e}")
        
        self.wait_for_everyone()
    
    def _load_lora_adapter(self, checkpoint_path: str):
        """Load LoRA adapter weights and configuration.
        
        Args:
            checkpoint_path: Directory containing the LoRA adapter
        """
        if not self._wrapped_models:
            return
        
        lora_load_path = os.path.join(checkpoint_path, "lora_adapter")
        
        # Check if LoRA adapter exists
        if not os.path.exists(lora_load_path):
            if self.is_main_process:
                print(f"LoRA adapter not found at {lora_load_path}, skipping load")
            return
        
        # Get the first wrapped model
        target_model = self._wrapped_models[0]
        unwrapped = self.unwrap_model(target_model)
        
        # Check if model has peft_config
        if not hasattr(unwrapped, "peft_config"):
            if self.is_main_process:
                print("Warning: Model does not have peft_config, skipping LoRA adapter load")
            return
        
        try:
            # Load LoRA weights
            safetensors_path = os.path.join(lora_load_path, "adapter_model.safetensors")
            bin_path = os.path.join(lora_load_path, "adapter_model.bin")
            
            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                lora_weights = load_file(safetensors_path)
            elif os.path.exists(bin_path):
                lora_weights = torch.load(bin_path, map_location="cpu")
            else:
                if self.is_main_process:
                    print(f"Warning: LoRA weights not found in {lora_load_path}")
                return
            
            # Load weights into model
            if isinstance(target_model, FullyShardedDataParallel):
                with FullyShardedDataParallel.summon_full_params(target_model, recurse=True):
                    # Filter to only LoRA parameters
                    current_state = target_model.state_dict()
                    lora_state = {k: v for k, v in lora_weights.items() if k in current_state}
                    target_model.load_state_dict(lora_state, strict=False)
            else:
                current_state = target_model.state_dict()
                lora_state = {k: v for k, v in lora_weights.items() if k in current_state}
                target_model.load_state_dict(lora_state, strict=False)
            
            if self.is_main_process:
                print(f"LoRA adapter loaded from: {lora_load_path}")
        
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Failed to load LoRA adapter: {e}")
        
        self.wait_for_everyone()
