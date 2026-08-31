import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
from pydantic import BaseModel
from typing import List, Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - fallback for environments without the package installed
    class OllamaLLM:
        def __init__(self, model: str, base_url: str = "http://localhost:11434"):
            self.model = model
            self.base_url = base_url

        def invoke(self, prompt: str) -> str:
            return f"[mock-fullstack] {self.model} responded to: {prompt[:120]}"

app = FastAPI()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model="phi3", base_url=OLLAMA_BASE_URL)


class TaskInput(BaseModel):
    task_id: str
    task_type: str = "fullstack"
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


@app.post("/run")
async def process_task(data: TaskInput):
    async def generator():
        logs = []
        def log(msg):
            logs.append(msg)
            return json.dumps({"type": "agent_thought", "text": msg}) + "\n"
        
        yield log("Analyzing fullstack requirements...")
        
        prompt = f'''
You are an expert Fullstack Developer Agent.
The user has requested the following task: {data.description}
Context: {data.context}

Please provide a complete implementation. If they asked for a full stack app (like a React frontend and Node backend), write out the main files required.
Output ONLY the code and necessary file structure in Markdown format. Do not include excessive conversational filler.
'''
        try:
            yield log("Generating code with LLM...")
            
            # Call the LLM in a background thread to avoid blocking the event loop
            generated_code = await asyncio.to_thread(llm.invoke, prompt)
            
            yield log("Code generation complete.")
            
            yield json.dumps({"type": "task_output", "output": dict(
                task_id=data.task_id,
                status="success",
                result=generated_code,
                summary="Successfully generated fullstack application code.",
                next_agent=None,
                logs=logs,
            )}) + "\n"
        except Exception as e:
            yield log(f"Error during LLM generation: {str(e)}")
            yield json.dumps({"type": "task_output", "output": {"task_id": data.task_id, "status": "error", "result": f"LLM Generation failed: {str(e)}"}}) + "\n"
        return
    
    return StreamingResponse(generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
