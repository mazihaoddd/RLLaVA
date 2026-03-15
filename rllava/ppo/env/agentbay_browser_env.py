import re
import os
from typing import Any, Dict, Optional, Tuple, Union
from agentbay import AgentBay, CreateSessionParams
from agentbay._common.models import BrowserOption
from agentbay._common.models.browser_agent import ActOptions, ObserveOptions
from .base import BaseEnv


class AgentBayBrowserEnv(BaseEnv):
    """AgentBay Browser environment for web interaction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("AGENTBAY_API_KEY")
        self.agent_bay = AgentBay(api_key=self.api_key) if self.api_key else None
        self.session = None
        self.start_url = config.get("start_url", "https://www.google.com")
        self.max_steps = config.get("max_steps", 20)
        self.step_count = 0

    def reset(self) -> str:
        """Initialize browser and return initial observation."""
        if not self.agent_bay:
            return "Error: AgentBay API key not configured"
        
        # Create session with browser image
        result = self.agent_bay.create(CreateSessionParams(image_id="browser_latest"))
        if not result.success:
            return f"Error: {result.error_message}"
        self.session = result.session
        
        # Initialize browser
        option = BrowserOption(headless=self.config.get("headless", True))
        if not self.session.browser.initialize(option):
            return "Error: Browser initialization failed"
        
        # Navigate to start URL
        self.session.browser.agent.navigate(self.start_url)
        self.step_count = 0
        
        # Return screenshot as observation
        return self._get_observation()

    def extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract action from model response. Returns None if not match."""
        # Pattern: <action type="navigate">url</action>
        m = re.search(r'<action\s+type="(\w+)">(.*?)</action>', content, re.DOTALL)
        if m:
            return {"type": m.group(1), "value": m.group(2).strip()}
        
        # Pattern: <browser_action>{"type": "...", "value": "..."}</browser_action>
        m = re.search(r'<browser_action>(.*?)</browser_action>', content, re.DOTALL)
        if m:
            import json
            try:
                return json.loads(m.group(1).strip())
            except:
                return None
        
        return None

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, dict]:
        """Execute action. Returns (obs, reward, done, info)."""
        if not self.session or not self.session.browser.is_initialized():
            return "Error: Browser not initialized", -1.0, True, {"error": "not_initialized"}
        
        self.step_count += 1
        action_type = action.get("type", "")
        value = action.get("value", "")
        
        try:
            if action_type == "navigate":
                self.session.browser.agent.navigate(value)
                reward = 1.0
            elif action_type == "act":
                result = self.session.browser.agent.act(ActOptions(action=value))
                reward = 1.0 if result.success else -0.5
            elif action_type == "observe":
                success, _ = self.session.browser.agent.observe(ObserveOptions(instruction=value))
                reward = 1.0 if success else 0.0
            elif action_type == "done":
                obs = self._get_observation()
                return obs, 1.0, True, {"reason": "task_complete"}
            else:
                return f"Unknown action type: {action_type}", -0.5, False, {}
            
            obs = self._get_observation()
            done = self.step_count >= self.max_steps
            info = {"step": self.step_count, "action": action_type}
            
            return obs, reward, done, info
            
        except Exception as e:
            return f"Error: {e}", -1.0, False, {"error": str(e)}

    def _get_observation(self) -> str:
        """Get current page observation (screenshot as base64)."""
        try:
            screenshot = self.session.browser.agent.screenshot()
            return f"[Screenshot: {screenshot[:100]}...]" if len(screenshot) > 100 else screenshot
        except:
            return "[Screenshot unavailable]"

    def close(self):
        """Close browser and session."""
        if self.session:
            try:
                self.session.browser.agent.close()
                self.agent_bay.delete(self.session)
            finally:
                self.session = None
