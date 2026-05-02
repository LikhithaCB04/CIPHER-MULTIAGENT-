from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests, json, os

app = FastAPI()

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from langchain_ollama import OllamaLLM
# Initialize the Brain Model (capable of multilingual understanding)
llm = OllamaLLM(model="llama3")

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

class Task(BaseModel):
    description: str
    context: str = ""

# Map of agents and their ports
AGENT_PORTS = {
    "data_science": "8001",
    "fullstack": "8002",
    "security": "8003",
    "devops": "8004",
    "ai_specialist": "8005"
}

@app.post('/chat')
async def chat_with_orchestrator(request: ChatRequest):
    """
    Multilingual Chat Endpoint.
    Understands English, Hindi, Kannada, Telugu.
    """
    prompt = f'''
    You are the Antigravity Orchestrator, an extremely intelligent multi-agent AI system.
    You must respond to the user in their requested language: {request.language}.
    You can understand English, Hindi, Kannada, and Telugu.
    
    User says: {request.message}
    '''
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"[MOCK RESPONSE]: Backend connected successfully! However, your local Ollama 'llama3' model is not running. Start Ollama to chat properly. (Error: {str(e)[:50]})"
        
    return {"response": response}

@app.post('/upload')
async def handle_file_upload(file: UploadFile = File(...)):
    """
    Handles Folders, Images, PDFs, Audio, Word Docs.
    """
    # Save the uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    return {"status": "success", "file_path": file_path, "message": f"Successfully processed {file.filename}"}

@app.post('/run')
def run_task(task: Task):
    # Ask the LLM to route the task to one or more of the 5 agents
    prompt = f'''
    You are a task router for a multi-agent AI system.
    Given this task: {task.description}
    Choose one or more agents from: data_science, fullstack, security, devops, ai_specialist
    Return ONLY a JSON list like: ["fullstack", "ai_specialist"]
    '''
    
    agents_raw = llm.invoke(prompt)
    try:
        # Simple extraction logic for JSON list
        if "[" in agents_raw and "]" in agents_raw:
            json_str = agents_raw[agents_raw.find("["):agents_raw.rfind("]")+1]
            agents = json.loads(json_str)
        else:
            agents = ["ai_specialist"]
    except:
        agents = ["ai_specialist"]

    results = []
    for agent in agents:
        port = AGENT_PORTS.get(agent, "8005")
        try:
            r = requests.post(f'http://localhost:{port}/run', 
                              json={'task_id': 't1', 'description': task.description, 'context': task.context})
            results.append(r.json())
        except Exception as e:
            results.append({"error": f"Agent {agent} is not running on port {port}. Please start it."})

    return {'agents_used': agents, 'results': results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)