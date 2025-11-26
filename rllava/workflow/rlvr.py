import asyncio
from collections import defaultdict
from typing import Any, Dict

from rllava.ppo.ppo import PPO


class RLVRWorkflow:
    """
    Encapsulate rollout into a workflow compatible interface.
    """

    def __init__(self, ppo: PPO):
        self.ppo = ppo

    async def arun_episode(self, engine: Any, data: Dict[str, Any]):
        """Asynchronously generate one rollout batch from a dataloader batch dict.

        Args:
            engine: placeholder for engine compatibility; not used for local rollout.
            data: a collated batch dict from dataloader.
        Returns:
            DataProto: rollout results including prompts, responses, masks, and rewards.
        """
        return await asyncio.to_thread(self.run_batch, engine, data)

    def run_batch(self, engine: Any, data: Dict[str, Any]):
        # Reuse RLLaVA's existing context that syncs weights into rollout engine
        with self.ppo.generate_context():
            return self.ppo.generate_one_batch(data)




