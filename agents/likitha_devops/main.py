from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.llms import Ollama

app = FastAPI()

# We use 'phi3' because it's fast and fits in your laptop's RAM
llm = Ollama(model="phi3")

# --- THE CONTRACT (MUST MATCH team's contract.json) ---
class TaskInput(BaseModel):
    task_id: str
    task_type: str
    description: str
    context: str
    priority: str

class TaskOutput(BaseModel):
    task_id: str
    status: str
    result: str
    summary: str
    next_agent: Optional[str] = None
    logs: List[str]

# --- THE AI LOGIC ---
@app.post("/run", response_model=TaskOutput)
async def run_devops_agent(task: TaskInput):
    print(f"⚙️  DevOps Agent is architecting infrastructure for task: {task.task_id}")

    prompt = f"""
    You are a Senior DevOps and Cloud Architect.
    USER REQUEST: {task.description}
    CONTEXT: {task.context}
    
    Please provide:
    1. A professional Dockerfile.
    2. A docker-compose.yml file.
    3. A brief explanation of the setup.
    
    Return the output clearly.
    """

    try:
        # AI generates the response
        ai_response = llm.invoke(prompt)
        
        return TaskOutput(
            task_id=task.task_id,
            status="success",
            result=ai_response,
            summary="Generated Docker and Compose configs using phi3.",
            next_agent=None, 
            logs=["Received task", "Invoked phi3", "Generated Configs"]
        )
    except Exception as e:
        return TaskOutput(
            task_id=task.task_id,
            status="error",
            result=f"Error: {str(e)}",
            summary="Failed to generate configs.",
            next_agent=None,
            logs=["Error occurred"]
        )

if __name__ == "__main__":
    import uvicorn
    # Port 8004 is Likitha's assigned port
    uvicorn.run(app, host="0.0.0.0", port=8004)