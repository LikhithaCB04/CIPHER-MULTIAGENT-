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
    allow_credentials=False,
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
llm = OllamaLLM(model="phi3", base_url=OLLAMA_BASE_URL)

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


# Map of agents to their docker-compose service name + the port each
# agent's own Dockerfile actually binds uvicorn to. Keep this in sync with
AGENT_SERVICES = {
    "data_science": {"host": os.environ.get("DATA_SCIENCE_HOST", "localhost"), "port": 8001},
    "fullstack": {"host": os.environ.get("FULLSTACK_HOST", "localhost"), "port": 8002},
    "security": {"host": os.environ.get("SECURITY_HOST", "localhost"), "port": 8003},
    "devops": {"host": os.environ.get("DEVOPS_HOST", "localhost"), "port": 8004},
    "ai_specialist": {"host": os.environ.get("AI_SPECIALIST_HOST", "localhost"), "port": 8005},
}

CONFIRMATION_STATE = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                if data.get("type") == "confirmation_response":
                    CONFIRMATION_STATE[data["task_id"]] = data["status"]
            except Exception:
                pass
    except Exception:
        connected_clients.discard(websocket)
        await websocket.close()

@app.get("/confirmations/{task_id}")
async def get_confirmation(task_id: str):
    return {"status": CONFIRMATION_STATE.get(task_id, "pending")}


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
        return ["devops"]
    if any(keyword in lowered for keyword in ["security", "audit", "vulnerability", "threat", "auth", "malware"]):
        return ["security", "devops"]
    if any(keyword in lowered for keyword in ["react", "frontend", "ui", "typescript", "vite", "api", "component"]):
        return ["fullstack"]
    if any(keyword in lowered for keyword in ["data", "analysis", "pandas", "ml", "model", "chart", "numpy"]):
        return ["data_science"]
    return ["fullstack"]


@app.post('/run')
async def run_task(task: Task):
    task_id = task.task_id or f"task-{uuid.uuid4().hex[:8]}"
    await broadcast({"event": "task_received", "task_id": task_id, "description": task.description})

    prompt = f'''
    You are a task router for a multi-agent AI system.
    Given this task: {task.description}
    If the task mentions a git URL, github, or repository, route it to ai_specialist.
    Otherwise choose one or more agents from: data_science, fullstack, security, devops.
    Do NOT route to ai_specialist unless a repository is explicitly mentioned.
    Return ONLY a JSON list like: ["fullstack"] or ["data_science", "devops"]
    '''

    try:
        if "github.com" in task.description.lower() or "http://" in task.description.lower() or "https://" in task.description.lower():
            agents = ["ai_specialist"]
        elif "data:image" in task.context.lower():
            agents = ["ai_specialist"]
        else:
            agents_raw = await asyncio.to_thread(llm.invoke, prompt)
            if "[" in agents_raw and "]" in agents_raw:
                json_str = agents_raw[agents_raw.find("["):agents_raw.rfind("]") + 1]
                agents = json.loads(json_str)
                if not isinstance(agents, list):
                    agents = []
            else:
                agents = choose_agents(task.description)
                
            if not agents:
                agents = choose_agents(task.description)
    except Exception:
        agents = choose_agents(task.description)

    import httpx
    results = []
    
    async with httpx.AsyncClient(timeout=900.0) as client:
        for index, agent in enumerate(agents):
            await broadcast({"event": "agent_started", "agent": agent, "task_id": task_id})
            service = AGENT_SERVICES.get(agent, AGENT_SERVICES["ai_specialist"])
            url = f"http://{service['host']}:{service['port']}"
            try:
                payload = {
                    "task_id": task_id,
                    "task_type": task.task_type,
                    "description": task.description,
                    "context": task.context,
                    "priority": task.priority,
                }
                
                final_response_data = None
                
                async with client.stream("POST", f"{url}/run", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("type") == "agent_thought":
                                await broadcast({
                                    "event": "agent_thought",
                                    "agent": agent,
                                    "task_id": task_id,
                                    "text": data.get("text", "")
                                })
                            elif data.get("type") == "confirmation_required":
                                await broadcast({
                                    "event": "confirmation_required",
                                    "agent": agent,
                                    "task_id": data.get("task_id"),
                                    "tool": data.get("tool"),
                                    "action": data.get("action")
                                })
                            elif data.get("type") == "task_output":
                                final_response_data = data.get("output", {})
                        except json.JSONDecodeError:
                            pass
                
                if not final_response_data:
                    final_response_data = {"result": "No final output returned by agent"}
                    
                next_agent = final_response_data.get("next_agent") or (agents[index + 1] if index + 1 < len(agents) else None)
                summary = final_response_data.get("summary") or final_response_data.get("result") or "Agent completed."
                results.append(final_response_data)
                
                await broadcast({
                    "event": "agent_finished",
                    "agent": agent,
                    "task_id": task_id,
                    "result_summary": summary,
                    "next_agent": next_agent,
                })
            except Exception as e:
                summary = f"Agent {agent} is not reachable at {url} or failed. Error: {str(e)}"
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
