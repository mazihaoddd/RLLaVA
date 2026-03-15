import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    step: int
    action: str
    observation: str = ""
    reward: float = 0.0
    done: bool = False
    screenshot_path: str = ""
    logprob: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop('logprob')
        d.pop('token_ids')
        return d


@dataclass 
class Trajectory:
    """Complete trajectory for a task."""
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    instruction: str = ""
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_reward: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "steps": [s.to_dict() for s in self.steps],
            "final_reward": self.final_reward,
            "created_at": self.created_at,
        }


class TrajectoryStorage:
    """Trajectory recorder and storage backend for workflow rollouts."""

    def __init__(self, root_dir: str = "trajectories"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.current_trajectory: Optional[Trajectory] = None

    def start_trajectory(self, task_id: str = "", instruction: str = "") -> str:
        """Start a new trajectory, return trajectory_id."""
        self.current_trajectory = Trajectory(task_id=task_id, instruction=instruction)
        traj_dir = self.root_dir / self.current_trajectory.trajectory_id
        traj_dir.mkdir(exist_ok=True)
        return self.current_trajectory.trajectory_id

    def add_step(
        self,
        step: int,
        action: str,
        observation: str = "",
        reward: float = 0.0,
        done: bool = False,
        screenshot: Any = None,
        logprob: torch.Tensor = None,
        token_ids: torch.Tensor = None,
    ):
        """Add one already-normalized step to the current trajectory."""
        if not self.current_trajectory:
            return

        traj_dir = self.root_dir / self.current_trajectory.trajectory_id
        screenshot_path = self._save_screenshot(traj_dir, step, screenshot)
        self._save_step_tensors(traj_dir, step, logprob, token_ids)

        self.current_trajectory.steps.append(
            TrajectoryStep(
                step=step,
                action=action,
                observation=observation,
                reward=float(reward),
                done=bool(done),
                screenshot_path=screenshot_path,
                logprob=logprob,
                token_ids=token_ids,
            )
        )

    def record_tool_step(
        self,
        *,
        step: int,
        action: str,
        tool_name: str,
        result,
        reward: float = 0.0,
        done: bool = False,
        logprob: torch.Tensor = None,
        token_ids: torch.Tensor = None,
    ) -> None:
        """Normalize and record a tool interaction step."""
        observation, screenshot = self._format_tool_result(tool_name, result)
        self.add_step(
            step=step,
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            screenshot=screenshot,
            logprob=logprob,
            token_ids=token_ids,
        )

    def record_environment_step(
        self,
        *,
        step: int,
        action: str,
        obs,
        info: dict,
        reward: float = 0.0,
        done: bool = False,
        logprob: torch.Tensor = None,
        token_ids: torch.Tensor = None,
    ) -> None:
        """Normalize and record an environment transition step."""
        screenshot = self._extract_screenshot(obs)
        observation = str(info) if screenshot is not None else f"{obs}\n{info}"
        self.add_step(
            step=step,
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            screenshot=screenshot,
            logprob=logprob,
            token_ids=token_ids,
        )

    def finish_trajectory(self, final_reward: float = 0.0) -> Optional[str]:
        """Finish and save current trajectory, return path."""
        if not self.current_trajectory:
            return None

        self.current_trajectory.final_reward = final_reward
        traj_dir = self.root_dir / self.current_trajectory.trajectory_id

        # Save trajectory metadata
        traj_path = traj_dir / "trajectory.json"
        with open(traj_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_trajectory.to_dict(), f, indent=2)

        path = str(traj_path)
        self.current_trajectory = None
        return path

    def load_trajectory(self, trajectory_id: str) -> Optional[Dict]:
        """Load a trajectory by id."""
        traj_path = self.root_dir / trajectory_id / "trajectory.json"
        if traj_path.exists():
            with open(traj_path, encoding='utf-8') as f:
                return json.load(f)
        return None

    def list_trajectories(self) -> List[str]:
        """List all trajectory ids."""
        return [d.name for d in self.root_dir.iterdir() if d.is_dir()]

    @staticmethod
    def _extract_screenshot(obs):
        if isinstance(obs, dict) and obs.get("screenshot") is not None:
            return obs["screenshot"]
        if hasattr(obs, "save") or isinstance(obs, (bytes, bytearray)):
            return obs
        return None

    @staticmethod
    def _format_tool_result(tool_name: str, result) -> tuple[str, Any]:
        """Convert raw tool outputs into a text observation plus optional image."""
        screenshot = None
        if isinstance(result, dict) and "image" in result:
            screenshot = result["image"]
            text = str(result.get("text", "")).strip()
            observation = f"[Tool: {tool_name}]\n<image>"
            if text:
                observation = f"{observation}\n{text}"
            return observation, screenshot

        if (
            hasattr(result, "save")
            or isinstance(result, (bytes, bytearray))
            or (isinstance(result, dict) and "bytes" in result)
        ):
            screenshot = result
            return f"[Tool: {tool_name}]\n<image>", screenshot

        return f"[Tool: {tool_name}]\n{str(result)}", None

    @staticmethod
    def _save_screenshot(traj_dir: Path, step: int, screenshot: Any) -> str:
        """Persist one screenshot artifact and return its file path."""
        if screenshot is None:
            return ""

        screenshot_path = traj_dir / f"step_{step}.png"
        if hasattr(screenshot, "save"):
            screenshot.save(screenshot_path)
        elif isinstance(screenshot, dict) and "bytes" in screenshot:
            with Image.open(BytesIO(screenshot["bytes"])) as image:
                image.save(screenshot_path)
        elif isinstance(screenshot, (bytes, bytearray)):
            with Image.open(BytesIO(screenshot)) as image:
                image.save(screenshot_path)
        else:
            return ""

        return str(screenshot_path)

    @staticmethod
    def _save_step_tensors(
        traj_dir: Path,
        step: int,
        logprob: torch.Tensor = None,
        token_ids: torch.Tensor = None,
    ) -> None:
        """Persist logprob / token ids for one step when available."""
        if logprob is None:
            return

        torch.save(
            {"logprob": logprob, "token_ids": token_ids},
            traj_dir / f"step_{step}.pt",
        )


class StepwiseSplitter:
    """Split trajectory into per-step training samples with reward reshaping."""
    
    def __init__(
        self,
        gamma: float = 0.99,           # discount factor for returns
        reward_mode: str = "terminal", # "terminal", "uniform", "shaped"
        success_bonus: float = 1.0,
        step_penalty: float = -0.01,
    ):
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
    
    def reshape_rewards(self, trajectory: Dict) -> List[float]:
        """Reshape final reward to per-step rewards."""
        n_steps = len(trajectory["steps"])
        final_reward = trajectory["final_reward"]
        
        if n_steps == 0:
            return []
        
        if self.reward_mode == "terminal":
            # Only last step gets reward
            rewards = [0.0] * (n_steps - 1) + [final_reward]
        
        elif self.reward_mode == "uniform":
            # Distribute reward uniformly
            rewards = [final_reward / n_steps] * n_steps
        
        elif self.reward_mode == "shaped":
            # Shaped: step penalty + terminal bonus
            rewards = [self.step_penalty] * n_steps
            if final_reward > 0:
                rewards[-1] += self.success_bonus
        
        else:
            rewards = [0.0] * n_steps
        
        return rewards
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns (reward-to-go) for each step."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def split(self, trajectory: Dict) -> List[Dict]:
        """Split trajectory into per-step training samples.
        
        Returns:
            List of dicts, each containing:
            - step: step index
            - action: action text
            - observation: observation text
            - screenshot_path: path to screenshot
            - reward: reshaped reward for this step
            - return_: discounted return from this step
            - logprob_path: path to logprob .pt file
        """
        steps = trajectory["steps"]
        if not steps:
            return []
        
        rewards = self.reshape_rewards(trajectory)
        returns = self.compute_returns(rewards)
        
        samples = []
        for i, step in enumerate(steps):
            sample = {
                "trajectory_id": trajectory["trajectory_id"],
                "task_id": trajectory["task_id"],
                "instruction": trajectory["instruction"],
                "step": step["step"],
                "action": step["action"],
                "observation": step["observation"],
                "screenshot_path": step["screenshot_path"],
                "reward": rewards[i],
                "return_": returns[i],
                "done": step["done"],
            }
            samples.append(sample)
        
        return samples
    
    def split_batch(self, trajectories: List[Dict]) -> List[Dict]:
        """Split multiple trajectories into training samples."""
        samples = []
        for traj in trajectories:
            samples.extend(self.split(traj))
        return samples


def load_and_split(
    storage: TrajectoryStorage,
    splitter: StepwiseSplitter,
    trajectory_ids: List[str] = None,
) -> List[Dict]:
    """Load trajectories and split into training samples."""
    if trajectory_ids is None:
        trajectory_ids = storage.list_trajectories()
    
    trajectories = []
    for tid in trajectory_ids:
        traj = storage.load_trajectory(tid)
        if traj:
            trajectories.append(traj)
    
    return splitter.split_batch(trajectories)
