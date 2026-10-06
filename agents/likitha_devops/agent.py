import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in __file__ else "../.env"))

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

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


class TaskInput(BaseModel):
    task_id: str
    task_type: str = "devops"
    description: str
    context: str = ""
    priority: Optional[str] = "medium"
    history: Optional[list] = None


class TaskOutput(BaseModel):
    task_id: str
    status: str
    result: str
    summary: str
    next_agent: Optional[str] = None
    logs: List[str]


@app.get('/health')
async def health_check():
    return {"status": "ok"}


from fastapi.responses import StreamingResponse
import json

@app.post("/run")
async def run_devops_task(task: TaskInput):
    async def generator():
        logs = []
        def log(msg):
            logs.append(msg)
            return json.dumps({"type": "agent_thought", "text": msg}) + "\n"
        
        yield log(f"Received devops task {task.task_id}")
        try:
            prompt = f"""
            You are a Senior DevOps and Cloud Architect.
            TASK: {task.description}
            CONTEXT: {task.context}

            Provide a concise deployment plan with a Dockerfile, docker-compose snippet,
            and CI recommendations.
            """
            response = llm.invoke(prompt)
            yield log("Prepared infrastructure guidance")
            yield json.dumps({"type": "task_output", "output": dict(
                task_id=task.task_id,
                status="success",
                result=response,
                summary="Generated a deployment-oriented response using the shared contract format.",
                next_agent=None,
                logs=logs,
            )}) + "\n"
        except Exception as exc:
            yield log(str(exc))
            yield json.dumps({"type": "task_output", "output": dict(
                task_id=task.task_id,
                status="error",
                result=f"DevOps agent failed: {exc}",
                summary="The deployment assistant could not complete the request.",
                next_agent=None,
                logs=logs,
            )}) + "\n"
    return StreamingResponse(generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
