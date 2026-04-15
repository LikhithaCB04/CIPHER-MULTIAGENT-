from langchain_ollama import OllamaLLM
from fastapi import FastAPI
from pydantic import BaseModel
import requests, json

app = FastAPI()

# This is the 'Brain'. It uses llama3 to decide which agent to use.
llm = OllamaLLM(model="llama3")

class Task(BaseModel):
    description: str
    context: str = ""

@app.post('/run')
def run_task(task: Task):
    # Step 1: Ask the LLM which agent should handle this
    prompt = f'''
    You are a task router for a multi-agent AI system.
    Given this task: {task.description}
    Choose one or more agents from: data_science, fullstack, security, devops
    Return ONLY a JSON list like: ["fullstack", "security"]
    '''
    
    # Get the decision from the AI
    agents_raw = llm.invoke(prompt)
    try:
        agents = json.loads(agents_raw)
    except:
        agents = ["fullstack"] # Fallback if AI doesn't return perfect JSON

    results = []
    for agent in agents:
        # Map the agent name to its specific port
        port_map = {
            "data_science": "8001",
            "fullstack": "8002",
            "security": "8003",
            "devops": "8004"
        }
        port = port_map.get(agent, "8002")
        
        # Try to call the agent's API
        try:
            r = requests.post(f'http://localhost:{port}/run', 
                              json={'task_id': 't1', 'task_type': agent, 
                                    'description': task.description, 'context': task.context})
            results.append(r.json())
        except Exception as e:
            results.append({"error": f"Agent {agent} is not running on port {port}. Please start the agent first."})

    return {'agents_used': agents, 'results': results}