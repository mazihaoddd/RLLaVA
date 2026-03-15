import re
import os
import json
from typing import Any, Dict, Optional, Tuple, Union
from agentbay import AgentBay, CreateSessionParams
from .base import BaseEnv


class AgentBayComputerEnv(BaseEnv):
    """AgentBay Computer environment for desktop GUI automation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("AGENTBAY_API_KEY")
        self.agent_bay = AgentBay(api_key=self.api_key) if self.api_key else None
        self.session = None
        self.max_steps = config.get("max_steps", 30)
        self.step_count = 0

    def reset(self) -> str:
        """Initialize computer environment and return initial observation."""
        if not self.agent_bay:
            return "Error: AgentBay API key not configured"
        
        result = self.agent_bay.create(CreateSessionParams(image_id="computer_latest"))
        if not result.success:
            return f"Error: {result.error_message}"
        self.session = result.session
        self.step_count = 0
        return self._get_observation()

    def extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract action from model response. Returns None if not match."""
        # Pattern: <action type="click" x="100" y="200"/>
        m = re.search(r'<action\s+type="(\w+)"([^>]*)(?:>(.*?)</action>|/>)', content, re.DOTALL)
        if m:
            action = {"type": m.group(1)}
            attrs = m.group(2)
            for attr in re.findall(r'(\w+)="([^"]*)"', attrs):
                action[attr[0]] = attr[1]
            if m.group(3):
                action["value"] = m.group(3).strip()
            return action
        
        # Pattern: <computer_action>{"type": "...", ...}</computer_action>
        m = re.search(r'<computer_action>(.*?)</computer_action>', content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except:
                return None
        return None

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, dict]:
        """Execute action. Returns (obs, reward, done, info)."""
        if not self.session:
            return "Error: Session not initialized", -1.0, True, {"error": "not_initialized"}
        
        self.step_count += 1
        action_type = action.get("type", "")
        
        try:
            reward = self._execute_action(action_type, action)
            obs = self._get_observation()
            done = self.step_count >= self.max_steps or action_type == "done"
            return obs, reward, done, {"step": self.step_count, "action": action_type}
        except Exception as e:
            return f"Error: {e}", -1.0, False, {"error": str(e)}

    def _execute_action(self, action_type: str, action: Dict[str, Any]) -> float:
        """Execute action and return reward."""
        computer = self.session.computer
        
        if action_type == "click":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            button = action.get("button", "left")
            result = computer.click_mouse(x, y, button)
            return 1.0 if result.success else -0.5
        
        elif action_type == "move":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            result = computer.move_mouse(x, y)
            return 1.0 if result.success else -0.5
        
        elif action_type == "drag":
            result = computer.drag_mouse(
                int(action.get("from_x", 0)), int(action.get("from_y", 0)),
                int(action.get("to_x", 0)), int(action.get("to_y", 0))
            )
            return 1.0 if result.success else -0.5
        
        elif action_type == "scroll":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            direction = action.get("direction", "up")
            amount = int(action.get("amount", 1))
            result = computer.scroll(x, y, direction, amount)
            return 1.0 if result.success else -0.5
        
        elif action_type == "type":
            result = computer.input_text(action.get("text", ""))
            return 1.0 if result.success else -0.5
        
        elif action_type == "key":
            keys = action.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            result = computer.press_keys(keys)
            return 1.0 if result.success else -0.5
        
        elif action_type == "done":
            return 1.0
        
        return -0.5  # Unknown action

    def _get_observation(self) -> str:
        """Get current screen observation."""
        try:
            result = self.session.computer.screenshot()
            if result.success and result.data:
                screenshot = result.data
                return f"[Screenshot: {screenshot[:100]}...]" if len(screenshot) > 100 else screenshot
        except:
            pass
        return "[Screenshot unavailable]"

    def close(self):
        """Close session."""
        if self.session:
            try:
                self.agent_bay.delete(self.session)
            finally:
                self.session = None
