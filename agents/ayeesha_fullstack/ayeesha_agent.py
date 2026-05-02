from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM

app = FastAPI()
llm = OllamaLLM(model="llama3")

class AgentInput(BaseModel):
    task_id: str
    description: str
    context: str = ""

GODLY_TEMPLATES = """
You have access to the following premium UI templates (Godly-inspired):
1. 'Broken Glass Card': A glassmorphism card with a shattered refraction effect.
2. 'Interactive ASCII': A hero section that translates hover coordinates into an ASCII art ripple.
3. 'Smooth Loader': A full-page loader with SVG path drawing and text reveal.
4. 'Thermal Effects': A background gradient that maps to mouse movement like a thermal camera.
5. 'Entrance Reveal': Text and images that reveal smoothly via a staggered Framer Motion mask.

When a user asks for a 'powerful attractive interactive animated cool entry', use these templates.
"""

@app.post("/run")
async def process_task(data: AgentInput):
    prompt = f"""
    SYSTEM: You are Ayeesha, the Expert Full-Stack Developer Agent. 
    You create state-of-the-art React/Next.js/Vite applications.
    {GODLY_TEMPLATES}
    
    TASK: {data.description}
    CONTEXT: {data.context}
    
    Write the code and provide the implementation details.
    RESPONSE:
    """
    response = llm.invoke(prompt)
    
    return {
        "task_id": data.task_id,
        "status": "success",
        "result": response.strip(),
        "agent": "fullstack"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)
