import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in __file__ else "../.env"))

from fastapi import FastAPI
import os

app = FastAPI()

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
            
    async def astream(self, prompt):
        import asyncio
        response = await asyncio.to_thread(self.invoke, prompt)
        yield response

llm = LLMProxy()
