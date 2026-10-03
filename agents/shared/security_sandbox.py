
app = FastAPI()

from groq import Groq
import os
import asyncio

class GroqProxy:
    def __init__(self, model_name, max_tokens):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_name
        self.max_tokens = max_tokens
        
    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
        
    async def astream(self, prompt):
        response = await asyncio.to_thread(self.invoke, prompt)
        yield response

llm = GroqProxy(model_name="openai/gpt-oss-120b", max_tokens=800)
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
