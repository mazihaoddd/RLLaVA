import os
from typing import Dict, Type, Union
from .inference.base import InferenceEngine
from .inference.config import VLLMConfig, SGLangConfig
from .train.base import TrainEngine


__all__ = [
    "VLLMConfig",
    "SGLangConfig",
]


ENGINE_REGISTRY: Dict[str, Type[Union[InferenceEngine, TrainEngine]]] = {}

def EngineFactory(engine_name: str=None) -> Type[Union[InferenceEngine, TrainEngine]]:
    """Factory function to create engine instances.
    
    Args:
        engine_name: Name of the engine to create (e.g., "vllm", "sglang")
        
    Returns:
        Engine class that can be instantiated
        
    Raises:
        ValueError: If engine_name is not registered
    """
    if engine_name is None:
        return ENGINE_REGISTRY["accelerator"]
    engine_name = engine_name.lower()
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(f"Engine '{engine_name}' is not registered. Available engines: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[engine_name]

def register_engine(name: str):
    """Decorator to register an engine class.
    
    Args:
        name: Name to register the engine under
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[Union[InferenceEngine, TrainEngine]]):
        if not issubclass(cls, InferenceEngine) and not issubclass(cls, TrainEngine):
            raise ValueError(f"Engine class {cls.__name__} must inherit from InferenceEngine or TrainEngine")
        ENGINE_REGISTRY[name.lower()] = cls
        return cls
    return decorator

# Automatically import engine modules to register them
def _import_engines():
    """Import all engine modules to register them."""
    engine_dir = os.path.dirname(__file__)
    for dir in ['train', 'inference']:
        for filename in os.listdir(os.path.join(engine_dir, dir)):
            if filename.endswith('.py') and filename not in ['__init__.py', 'inference.py']:
                module_name = filename[:-3]  # Remove .py
                try:
                    __import__(f"rllava.engine.{dir}.{module_name}")
                except ImportError as e:
                    print(f"Warning: Could not import {dir}.{module_name}: {e}")

# Import engines when module is loaded
_import_engines()
