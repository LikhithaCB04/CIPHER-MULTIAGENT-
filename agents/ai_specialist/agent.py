from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - fallback for environments without the package installed
    class OllamaLLM:
        def __init__(self, model: str):
            self.model = model

        def invoke(self, prompt: str) -> str:
            return f"[mock] {self.model} responded to: {prompt[:120]}"

app = FastAPI()

# Initialize the model (llama3 or your preferred local model)
llm = OllamaLLM(model="llama3")


class Task(BaseModel):
    task_id: str
    task_type: str = "orchestrate"
    description: str
    context: str = ""
    priority: Optional[str] = "medium"


@app.post("/run")
async def process_task(data: Task):
    adaptive_prompt = f"""
    SYSTEM: You are the AI Specialist Agent. You are highly intelligent, analytical, and an expert in AI models and fine-tuning.
    TASK: {data.description}
    CONTEXT: {data.context}
    PRIORITY: {data.priority}

    RESPONSE:
    """
    response = llm.invoke(adaptive_prompt)

    return {
        "task_id": data.task_id,
        "status": "success",
        "result": response.strip(),
        "summary": "AI specialist analysis completed.",
        "next_agent": None,
        "logs": ["AI specialist reviewed the request and produced a structured response."],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8005)
