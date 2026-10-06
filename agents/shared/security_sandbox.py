import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in __file__ else "../.env"))


app = FastAPI()

from groq import Groq
import os
import asyncio

class LLMProxy:
    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct", max_tokens=2500):
        self.model_name = model_name
        self.api_key = os.environ.get("HF_API_KEY", "")
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=self.api_key)
        
    def invoke(self, prompt: str) -> str:
        try:
            res = self.client.chat_completion([{"role": "user", "content": prompt}], model=self.model_name)
            return res.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

llm = LLMProxy(model_name=os.environ.get("GROQ_AGENT_MODEL", "openai/gpt-oss-120b"), max_tokens=800)
import uuid
import time
import httpx
import json
import os

ORCHESTRATOR_URL = os.environ.get("VITE_API_URL", "http://orchestrator:8000")

def require_confirmation(tool_name: str, action_details: str, yield_fn=None, timeout: int = 120) -> bool:
    task_id = f"sec-{uuid.uuid4().hex[:6]}"
    
    # 1. Yield event out to orchestrator via the generator callback
    if yield_fn:
        msg = json.dumps({
            "type": "confirmation_required",
            "task_id": task_id,
            "tool": tool_name,
            "action": action_details
        }) + "\n"
        yield_fn(msg)
    
    # 2. Wait for confirmation via polling
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = httpx.get(f"{ORCHESTRATOR_URL}/confirmations/{task_id}", timeout=5.0)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "approved":
                    return True
                elif data.get("status") == "denied":
                    return False
        except Exception:
            pass
        time.sleep(2)
        
    return False # Fail closed on timeout
