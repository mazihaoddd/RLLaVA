import re
import os
import json
from typing import Any, Dict, Optional, Tuple, Union
from agentbay import AgentBay, CreateSessionParams
from .base import BaseEnv


class AgentBayMobileEnv(BaseEnv):
    """AgentBay Mobile environment for mobile device automation."""
    
    # Key codes for mobile device
    KEY_HOME = 3
    KEY_BACK = 4
    KEY_VOLUME_UP = 24
    KEY_VOLUME_DOWN = 25
    KEY_POWER = 26
    KEY_MENU = 82
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("AGENTBAY_API_KEY")
        self.agent_bay = AgentBay(api_key=self.api_key) if self.api_key else None
        self.session = None
        self.max_steps = config.get("max_steps", 30)
        self.step_count = 0

    def reset(self) -> str:
        """Initialize mobile environment and return initial observation."""
        if not self.agent_bay:
            return "Error: AgentBay API key not configured"
        
        result = self.agent_bay.create(CreateSessionParams(image_id="mobile_latest"))
        if not result.success:
            return f"Error: {result.error_message}"
        self.session = result.session
        self.step_count = 0
        return self._get_observation()

    def extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract action from model response. Returns None if not match."""
        # Pattern: <action type="tap" x="100" y="200"/>
        m = re.search(r'<action\s+type="(\w+)"([^>]*)(?:>(.*?)</action>|/>)', content, re.DOTALL)
        if m:
            action = {"type": m.group(1)}
            attrs = m.group(2)
            for attr in re.findall(r'(\w+)="([^"]*)"', attrs):
                action[attr[0]] = attr[1]
            if m.group(3):
                action["value"] = m.group(3).strip()
            return action
        
        # Pattern: <mobile_action>{"type": "...", ...}</mobile_action>
        m = re.search(r'<mobile_action>(.*?)</mobile_action>', content, re.DOTALL)
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
        mobile = self.session.mobile
        
        if action_type == "tap":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            result = mobile.tap(x, y)
            return 1.0 if result.success else -0.5
        
        elif action_type == "swipe":
            result = mobile.swipe(
                int(action.get("start_x", 0)), int(action.get("start_y", 0)),
                int(action.get("end_x", 0)), int(action.get("end_y", 0)),
                int(action.get("duration_ms", 300))
            )
            return 1.0 if result.success else -0.5
        
        elif action_type == "type":
            result = mobile.input_text(action.get("text", ""))
            return 1.0 if result.success else -0.5
        
        elif action_type == "key":
            key_name = action.get("key", "").upper()
            key_map = {
                "HOME": self.KEY_HOME, "BACK": self.KEY_BACK,
                "VOLUME_UP": self.KEY_VOLUME_UP, "VOLUME_DOWN": self.KEY_VOLUME_DOWN,
                "POWER": self.KEY_POWER, "MENU": self.KEY_MENU
            }
            key_code = key_map.get(key_name, int(action.get("key", 0)))
            result = mobile.send_key(key_code)
            return 1.0 if result.success else -0.5
        
        elif action_type == "start_app":
            result = mobile.start_app(action.get("cmd", ""), action.get("activity", ""))
            return 1.0 if result.success else -0.5
        
        elif action_type == "stop_app":
            result = mobile.stop_app_by_cmd(action.get("cmd", ""))
            return 1.0 if result.success else -0.5
        
        elif action_type == "done":
            return 1.0
        
        return -0.5  # Unknown action

    def _get_observation(self) -> str:
        """Get current screen observation."""
        try:
            result = self.session.mobile.screenshot()
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
