from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import requests, json, os, uuid


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from groq import Groq
import os
import asyncio

class LLMProxy:
    def __init__(self, model_name="gemini-3.1-pro", max_tokens=2500):
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
    def invoke(self, prompt: str) -> str:
        import requests, os
        payload = {"contents": [{"parts":[{"text": prompt}]}]}
        try:
            resp = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            if resp.status_code != 429:
                try:
                    data = resp.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                except: pass
            
            # Fallback to Cohere if Gemini quota is exhausted
            cohere_key = os.environ.get("COHERE_API_KEY", "")
            if cohere_key:
                co_res = requests.post("https://api.cohere.com/v1/chat",
                                       headers={"Authorization": f"Bearer {cohere_key}", "Content-Type": "application/json"},
                                       json={"model": "command-a-03-2025", "message": prompt})
                if co_res.status_code == 200:
                    return co_res.json()["text"]

            return f"Error: Gemini quota exhausted and fallback failed. {resp.text}"
        except Exception as e:
            return f"Error: {str(e)}"

llm = LLMProxy(model_name=os.environ.get("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b"), max_tokens=300)

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
    history: Optional[list] = None


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
        response = f"[MOCK RESPONSE]: Backend connected successfully! However, your Groq model failed to run. (Error: {str(e)[:50]})"

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

    # Strict file routing
    is_data_file = False
    if task.context:
        low_ctx = task.context.lower()
        if "csv" in low_ctx or "spreadsheetml" in low_ctx or "excel" in low_ctx or ".xls" in low_ctx:
            is_data_file = True
    try:
        if is_data_file:
            agents = ["data_science"]
        elif "github.com" in task.description.lower() or "http://" in task.description.lower() or "https://" in task.description.lower():
            agents = ["ai_specialist"]
        elif "data:image" in task.context.lower():
            agents = ["ai_specialist"]
        else:
            # Explicit mentions override LLM
            mentioned = []
            low_desc = task.description.lower()
            
            # Use regex to find order of mentions to keep the pipeline sequential
            agent_keywords = {
                "data_science": ["data science", "data_science", "dummy content", "data agent"],
                "fullstack": ["fullstack", "ui", "web page", "website", "frontend", "react"],
                "security": ["security", "audit"],
                "devops": ["devops", "deploy", "pipeline"]
            }
            
            # Find earliest occurrence of any keyword for each agent
            positions = {}
            for ag, kws in agent_keywords.items():
                pos = -1
                for kw in kws:
                    idx = low_desc.find(kw)
                    if idx != -1 and (pos == -1 or idx < pos):
                        pos = idx
                if pos != -1:
                    positions[ag] = pos
            
            if positions:
                # Sort agents by where they were mentioned in the text
                agents = [ag for ag, _ in sorted(positions.items(), key=lambda x: x[1])]
            else:
                prompt = f'''
                You are a task router for a multi-agent AI system.
                Given this task: {task.description}
                If the task mentions a git URL, github, or repository, route it to ai_specialist.
                If the task is a general greeting, small talk, or doesn't clearly require data analysis, code generation, security review, or deployment work, default to fullstack.
                Otherwise choose one or more agents from: data_science, fullstack, security, devops.
                Do NOT route to ai_specialist unless a repository is explicitly mentioned.
                CRITICAL: Return ONLY a valid JSON array of strings, no markdown, no backticks, no explanation. Example: ["fullstack"] or ["data_science", "devops"]
                '''
                agents_raw = await asyncio.to_thread(llm.invoke, prompt)
                if "[" in agents_raw and "]" in agents_raw:
                    json_str = agents_raw[agents_raw.find("["):agents_raw.rfind("]") + 1]
                    agents = json.loads(json_str)
                    if not isinstance(agents, list):
                        agents = []
                else:
                    agents = ["fullstack"]
                
                if not agents:
                    agents = ["fullstack"]
    except Exception:
        agents = ["fullstack"]

    import httpx
    results = []
    has_error = False
    
    # Clarification Check
    if len(task.description) < 100 and "Clarify:" not in task.description and not task.history:
        clarify_prompt = f"The user asked to '{task.description}'. This is too brief. Ask 3 clarifying questions to gather requirements, and give a blank space or options for them to fill. Format beautifully in Markdown."
        clarification = llm.invoke(clarify_prompt)
        await broadcast({
            "event": "agent_finished",
            "agent": "ai_specialist",
            "task_id": task_id,
            "result_summary": clarification,
            "next_agent": None
        })
        return {
            "status": "success",
            "task_id": task_id,
            "agents_used": ["ai_specialist"],
            "results": [{
                "task_id": task_id,
                "status": "success",
                "summary": "Please clarify your request:",
                "result": clarification
            }]
        }

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
                    "history": task.history,
                }
                
                resp = await client.post(f"{url}/run", json=payload)
                resp.raise_for_status()
                text = resp.text.strip()
                import json
                try:
                    lines = [line for line in text.split('\n') if line.strip()]
                    if lines:
                        parsed = json.loads(lines[-1])
                        data = parsed.get("output", parsed) if isinstance(parsed, dict) else parsed
                    else:
                        data = {"status": "error", "summary": "Empty output", "result": ""}
                except Exception as ex:
                    data = {"status": "error", "summary": "JSON error", "result": str(ex) + " on text: " + text[:50]}
                
                summary = data.get("summary", "Agent completed successfully.")
                if data.get("status") == "error":
                    has_error = True
                
                next_agent = None
                if index < len(agents) - 1:
                    next_agent = agents[index + 1]
                    
                results.append(data)
                
                await broadcast({
                    "event": "agent_finished",
                    "agent": agent,
                    "task_id": task_id,
                    "result_summary": summary,
                    "next_agent": next_agent,
                })
            except Exception as e:
                has_error = True
                summary = f"Agent {agent} is not reachable at {url} or failed. Error: {str(e)}"
                results.append({"error": summary, "task_id": task_id, "status": "error"})
                await broadcast({
                    "event": "agent_finished",
                    "agent": agent,
                    "task_id": task_id,
                    "result_summary": summary,
                    "next_agent": None,
                })

    return {
        "status": "error" if has_error else "success",
        "task_id": task_id,
        "agents_used": agents,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
