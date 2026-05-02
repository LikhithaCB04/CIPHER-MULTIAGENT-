from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM

app = FastAPI()

# Initialize the model (llama3 or your preferred local model)
llm = OllamaLLM(model="llama3")

class AgentInput(BaseModel):
    task_id: str
    description: str
    context: str = ""

@app.post("/run")
async def process_task(data: AgentInput):
    adaptive_prompt = f"""
    SYSTEM: You are the AI Specialist Agent. You are highly intelligent, analytical, and an expert in AI models and fine-tuning.
    TASK: {data.description}
    CONTEXT: {data.context}
    
    RESPONSE:
    """
    response = llm.invoke(adaptive_prompt)
    
    return {
        "task_id": data.task_id,
        "status": "success",
        "result": response.strip(),
        "agent": "ai_specialist"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8005)
