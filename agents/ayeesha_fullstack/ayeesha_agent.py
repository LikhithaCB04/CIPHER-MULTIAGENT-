from langchain_ollama import OllamaLLM
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
llm = OllamaLLM(model="codellama")

class Task(BaseModel):
    task_id: str
    task_type: str = "fullstack"
    description: str
    context: str = ''
    priority: str = "medium"

@app.post('/run')
def run(task: Task):
    prompt = f'''
    You are a senior full stack developer.
    Task: {task.description}
    Additional context: {task.context}
    
    Generate complete, working code. Include:
    1. Frontend component (React or plain HTML/CSS/JS)
    2. Backend API route (FastAPI)
    3. Database schema if needed (SQLAlchemy)
    4. One unit test
    
    Format each section clearly with comments.
    '''
    code = llm.invoke(prompt)
    
    return {
        'task_id': task.task_id,
        'status': 'success',
        'result': code,
        'summary': 'Generated full stack code',
        'next_agent': 'security',
        'logs': ['Code generated successfully']
    }