# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__file__)

# Environment class registry
ENV_REGISTRY = {
    "AgentBayBrowserEnv": "rllava.ppo.env.agentbay_browser_env.AgentBayBrowserEnv",
    "AgentBayComputerEnv": "rllava.ppo.env.agentbay_computer_env.AgentBayComputerEnv",
    "AgentBayMobileEnv": "rllava.ppo.env.agentbay_mobile_env.AgentBayMobileEnv",
    "OSWorldEnv": "rllava.ppo.env.osworld_env.OSWorldEnv",
    "OSWorldSubprocessEnv": "rllava.ppo.env.osworld_subprocess_env.OSWorldSubprocessEnv",
    "AndroidWorldEnv": "rllava.ppo.env.androidworld_env.AndroidWorldEnv",
    "AndroidWorldSubprocessEnv": "rllava.ppo.env.androidworld_subprocess_env.AndroidWorldSubprocessEnv",
}


def get_env_class(class_name: str):
    """Get environment class by name.
    
    Args:
        class_name: Either short name (e.g., 'OSWorldEnv') or full path
    
    Returns:
        Environment class
    """
    # Check if it's a registered short name
    if class_name in ENV_REGISTRY:
        full_path = ENV_REGISTRY[class_name]
    else:
        full_path = class_name
    
    # Import the class
    module_path, cls_name = full_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def initialize_env_from_config(env_config_file):
    """Initialize environment from configuration file.

    Args:
        env_config_file: Path to the environment configuration file.

    Returns:
        BaseEnv: Environment instance.
    """
    config = OmegaConf.load(env_config_file)
    
    # Get env class name and config
    cls_name = config.get("class_name", config.get("env_class"))
    if cls_name is None:
        raise ValueError("Environment config must specify 'class_name' or 'env_class'")
    
    env_cls = get_env_class(cls_name)
    env_config = OmegaConf.to_container(config.get("config", {}), resolve=True)
    
    env = env_cls(config=env_config)
    logger.info(f"Initialized environment '{cls_name}'")
    
    return env
