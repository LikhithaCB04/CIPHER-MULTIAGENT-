import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

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


llm = GroqProxy(model_name="llama-3.1-70b-versatile", max_tokens=800)


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
