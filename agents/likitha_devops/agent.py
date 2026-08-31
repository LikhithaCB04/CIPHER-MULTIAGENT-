import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

try:
    from langchain_community.llms import Ollama
except ImportError:  # pragma: no cover - fallback for environments without the package installed
    class Ollama:
        def __init__(self, model: str, base_url: str = "http://localhost:11434"):
            self.model = model
            self.base_url = base_url

        def invoke(self, prompt: str) -> str:
            return f"[mock-devops] {self.model} responded to: {prompt[:120]}"

app = FastAPI()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
llm = Ollama(model="phi3", base_url=OLLAMA_BASE_URL)


class TaskInput(BaseModel):
    task_id: str
    task_type: str = "devops"
    description: str
    context: str = ""
    priority: Optional[str] = "medium"


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
