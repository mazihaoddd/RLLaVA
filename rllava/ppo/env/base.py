from typing import Any, Dict, Optional, Tuple, Union


class BaseEnv:
    """Base class for environments. Uses standard RL interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def reset(self, task_config: Optional[Dict[str, Any]] = None) -> Any:
        """Reset environment. Returns initial observation.
        
        Args:
            task_config: Optional task-specific configuration dict.
                         For environments like OSWorld, this contains task id,
                         instruction, setup config, evaluator config, etc.
        
        Returns:
            obs: Can be Dict (e.g. {"screenshot": Image, "text": str}),
                 Image, str, or any format suitable for the task.
        """
        raise NotImplementedError
    
    def extract_action(self, content: str) -> Any:
        """Extract action from model response. Return None if not match."""
        raise NotImplementedError
        
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """Execute action.
        
        Returns:
            obs: Observation (Dict, Image, str, etc.)
            reward: Format reward for action execution
            done: Whether episode is finished
            info: Additional information
        """
        raise NotImplementedError
    
    def close(self):
        pass
