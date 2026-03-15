import re
import os
from typing import Dict, Any, Optional, Tuple
from agentbay import AgentBay, CreateSessionParams
from .base import BaseTool


class AgentBayCodeTool(BaseTool):
    """AgentBay云代码执行工具"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("AGENTBAY_API_KEY")
        self.agent_bay = AgentBay(api_key=self.api_key) if self.api_key else None
        self.session = None
        self.default_language = config.get("default_language", "python")
    
    def extract_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """提取代码和语言，返回None表示不匹配"""
        # 模式1：<code_execution language="python">code</code_execution>
        m = re.search(r'<code_execution(?:\s+language="([^"]+)")?\s*>(.*?)</code_execution>', content, re.DOTALL)
        if m:
            return {"language": m.group(1) or self.default_language, "code": m.group(2).strip()}
        
        # 模式2：```language\ncode\n```
        m = re.search(r'```([a-zA-Z]+)?\n(.*?)```', content, re.DOTALL)
        if m:
            return {"language": m.group(1) or self.default_language, "code": m.group(2).strip()}
        
        return None
    
    def execute(self, tool_content: Dict[str, Any]) -> Tuple[str, bool]:
        """执行代码，返回(结果, 是否成功)"""
        language = tool_content.get("language", self.default_language)
        code = tool_content.get("code", "")
        
        # run_code只支持python和javascript
        if language not in ("python", "javascript"):
            return f"Error: Unsupported language '{language}'", False
        
        try:
            if self.session is None:
                if not self.agent_bay:
                    return "Error: AgentBay API key not configured", False
                result = self.agent_bay.create(CreateSessionParams(image_id="code_latest"))
                if not result.success:
                    return f"Error: {result.error_message}", False
                self.session = result.session
            
            r = self.session.code.run_code(code, language)
            if not r.success:
                return f"Error: {getattr(r, 'error_message', 'Unknown error')}", False
            
            # 收集输出
            parts = []
            if r.logs:
                if r.logs.stdout:
                    parts.extend(r.logs.stdout)
                if r.logs.stderr:
                    parts.append(f"STDERR: {' '.join(r.logs.stderr)}")
            if r.results:
                for res in r.results:
                    if res.text:
                        parts.append(res.text)
            if r.error:
                return f"{r.error.name}: {r.error.value}", False
            
            return "\n".join(parts) if parts else "Execution completed", True
        except Exception as e:
            return f"Error: {e}", False
    
    def release(self):
        if self.session:
            try:
                self.agent_bay.delete(self.session)
            finally:
                self.session = None
