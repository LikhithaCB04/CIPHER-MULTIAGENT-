import json
import requests
import fitz
from PIL import Image
import pytesseract
import io
from fastapi import FastAPI, UploadFile, File, Form
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

app = FastAPI()

llm = OllamaLLM(model="llama3")

AGENT_PORTS = {
    "data_science": "8001",
    "fullstack": "8002",
    "security": "8003",
    "devops": "8004"
}

@app.post('/run')
async def master_orchestrator(
    description: str = Form(...), 
    file: UploadFile = File(None)
):
    context_text = ""

    if file:
        file_content = await file.read()
        
        if file.content_type == "application/pdf":
            doc = fitz.open(stream=file_content, filetype="pdf")
            context_text = "\n".join([page.get_text() for page in doc])
            
        elif file.content_type in ["image/png", "image/jpeg"]:
            image = Image.open(io.BytesIO(file_content))
            context_text = pytesseract.image_to_string(image)

    brain_prompt = f"""
    You are the Master Brain of a Multi-Agent Tech Team.
    USER TASK: "{description}"
    FILE CONTEXT: "{context_text[:1000]}"

    GOAL:
    1. If the task is a simple question (e.g., "What is Python?"), answer it directly.
    2. If the task requires technical work, select the best agents:
       - 'data_science': For analysis, ML, and data plots.
       - 'fullstack': For building UI, APIs, and features.
       - 'security': For auditing code and fixing vulnerabilities.
       - 'devops': For Docker, deployment, and CI/CD.

    YOU MUST RETURN ONLY A JSON OBJECT:
    {{
        "is_simple": true/false,
        "direct_answer": "your answer if simple, else empty",
        "selected_agents": ["agent_name1", "agent_name2"]
    }}
    """

    raw_decision = llm.invoke(brain_prompt)
    
    try:
        json_start = raw_decision.find('{')
        json_end = raw_decision.rfind('}') + 1
        decision = json.loads(raw_decision[json_start:json_end])
    except:
        return {"error": "Brain failed to format JSON", "raw": raw_decision}

    if decision["is_simple"]:
        return {"final_output": decision["direct_answer"]}

    agent_results = {}
    current_context = context_text

    for agent in ["fullstack"]:
        port = AGENT_PORTS.get(agent)
        if port:
            try:
                response = requests.post(
                    f"http://localhost:{port}/run",
                    json={
                        "task_id": "live_demo_001",
                        "task_type": agent,
                        "description": description,
                        "context": current_context
                    },
                    timeout=60
                )
                res_data = response.json()
                agent_results[agent] = res_data
                
                if "result" in res_data:
                    current_context += f"\n\nOutput from {agent}:\n{res_data['result']}"
            
            except Exception as e:
                agent_results[agent] = {"error": f"Agent on port {port} not reachable."}

    return {
        "summary": "Technical pipeline completed.",
        "agents_executed": list(agent_results.keys()),
        "full_technical_output": agent_results
    }