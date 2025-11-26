import torch
from typing import Dict, Optional, Union, Any
from accelerate import Accelerator
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin, DeepSpeedPlugin
from ...data.protocol import DataProto
from ...utils.logging import print_rank0
from contextlib import nullcontext


class TrainEngine:
    """Base class for all training engines.
    
    This class defines the common interface that all training engines must implement.
    Engines are responsible for managing the training process for PPO.
    """
    
    def __init__(self, config):
        self.config = config

    @property
    def world_size(self):
        raise NotImplementedError("world_size is not implemented")
    
    @property
    def rank(self):
        raise NotImplementedError("rank is not implemented")
    
    @property
    def num_processes(self):
        raise NotImplementedError("num_processes is not implemented")
    
    @property
    def is_main_process(self):
        raise NotImplementedError("is_main_process is not implemented")
    
    def wait_for_everyone(self):
        raise NotImplementedError("wait_for_everyone is not implemented")

    def prepare(self, *args, **kwargs):
        raise NotImplementedError("prepare is not implemented")

    def get_init_weight_context(self, use_meta_tensor=True):
        raise NotImplementedError("get_init_weight_context is not implemented")
    
    def unwrap_model(self, model):
        raise NotImplementedError("unwrap_model is not implemented")
    
    def unwrap_model_for_generation(self, model, is_peft_model):
        raise NotImplementedError("unwrap_model_for_generation is not implemented")
    
    def load_state(self, checkpoint_path):
        raise NotImplementedError("load_state is not implemented")
    
    def save_state(self, checkpoint_path):
        raise NotImplementedError("save_state is not implemented")
    
    def backward(self, loss):
        raise NotImplementedError("backward is not implemented")
    
    def clip_grad_norm_(self, parameters, max_norm):
        raise NotImplementedError("clip_grad_norm_ is not implemented")
    
    def get_model_weights(self, model=None):
        raise NotImplementedError("get_model_weights is not implemented")
    