from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import requests, json, os, uuid

app = FastAPI()

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - fallback for environments without the package installed
    class OllamaLLM:
        def __init__(self, model: str, base_url: str = "http://localhost:11434"):
            self.model = model
            self.base_url = base_url

        def invoke(self, prompt: str) -> str:
            return f"[mock] routed by {self.model}: {prompt[:80]}"

# Initialize the Brain Model (capable of multilingual understanding)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model="llama3", base_url=OLLAMA_BASE_URL)

connected_clients = set()


class ChatRequest(BaseModel):
    message: str
    language: str = "English"


class Task(BaseModel):
    description: str
    context: str = ""
    task_id: Optional[str] = None
    task_type: Optional[str] = "orchestrate"
    priority: Optional[str] = "medium"


# Map of agents and their ports
AGENT_PORTS = {
    "deepthi": "8001",
    "ayeesha": "8002",
    "mahima": "8003",
    "likitha": "8004",
    "ai_specialist": "8005",
}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.discard(websocket)
        await websocket.close()


async def broadcast(event: dict):
    dead = set()
    for ws in list(connected_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


@app.get('/health')
async def health_check():
    return {"status": "ok"}


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
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"status": "success", "file_path": file_path, "message": f"Successfully processed {file.filename}"}


def choose_agents(description: str):
    lowered = description.lower()
    if any(keyword in lowered for keyword in ["clone", "repo", "repository", "git", "fix bug", "run tests", "test suite", "pytest", "npm test"]):
        return ["ai_specialist"]
    if any(keyword in lowered for keyword in ["deploy", "docker", "kubernetes", "infra", "server", "ci/cd"]):
        return ["likitha"]
    if any(keyword in lowered for keyword in ["security", "audit", "vulnerability", "threat", "auth", "malware"]):
        return ["mahima", "likitha"]
    if any(keyword in lowered for keyword in ["react", "frontend", "ui", "typescript", "vite", "api", "component"]):
        return ["ayeesha", "ai_specialist"]
    if any(keyword in lowered for keyword in ["data", "analysis", "pandas", "ml", "model", "chart", "numpy"]):
        return ["deepthi", "ai_specialist"]
    return ["ai_specialist"]


@app.post('/run')
async def run_task(task: Task):
    task_id = task.task_id or f"task-{uuid.uuid4().hex[:8]}"
    await broadcast({"event": "task_received", "task_id": task_id, "description": task.description})

    prompt = f'''
    You are a task router for a multi-agent AI system.
    Given this task: {task.description}
    If the task mentions cloning a repo, a git URL, fixing a bug, or running tests,
    route it to ai_specialist. Otherwise choose one or more agents from:
    deepthi, ayeesha, mahima, likitha, ai_specialist.
    Return ONLY a JSON list like: ["ai_specialist"] or ["ayeesha", "ai_specialist"]
    '''

    try:
        agents_raw = llm.invoke(prompt)
        if "[" in agents_raw and "]" in agents_raw:
            json_str = agents_raw[agents_raw.find("["):agents_raw.rfind("]") + 1]
            agents = json.loads(json_str)
        else:
            agents = choose_agents(task.description)
    except Exception:
        agents = choose_agents(task.description)

    results = []
    for index, agent in enumerate(agents):
        await broadcast({"event": "agent_started", "agent": agent, "task_id": task_id})
        port = AGENT_PORTS.get(agent, "8005")
        try:
            payload = {
                "task_id": task_id,
                "task_type": task.task_type,
                "description": task.description,
                "context": task.context,
                "priority": task.priority,
            }
            response = await asyncio.to_thread(
                requests.post,
                f'http://localhost:{port}/run',
                json=payload,
                timeout=40,
            )
            response_data = await asyncio.to_thread(response.json)
            next_agent = response_data.get("next_agent") or (agents[index + 1] if index + 1 < len(agents) else None)
            summary = response_data.get("summary") or response_data.get("result") or "Agent completed."
            results.append(response_data)
            await broadcast({
                "event": "agent_finished",
                "agent": agent,
                "task_id": task_id,
                "result_summary": summary,
                "next_agent": next_agent,
            })
        except Exception as e:
            summary = f"Agent {agent} is not running on port {port}. Please start it."
            results.append({"error": summary, "task_id": task_id})
            await broadcast({
                "event": "agent_finished",
                "agent": agent,
                "task_id": task_id,
                "result_summary": summary,
                "next_agent": None,
            })

    await broadcast({"event": "pipeline_complete", "task_id": task_id})
    return {'task_id': task_id, 'agents_used': agents, 'results': results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)