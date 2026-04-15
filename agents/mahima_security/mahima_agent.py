from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
import subprocess
import tempfile
import os

app = FastAPI()
llm = Ollama(model="mahima-agent")

class Task(BaseModel):
    task_id: str
    description: str
    context: str 

# --- NEW: KNOWLEDGE BASE READER ---
def load_knowledge_base():
    knowledge_text = ""
    kb_path = "agents/mahima_security/knowledge_base/"
    
    if os.path.exists(kb_path):
        for filename in os.listdir(kb_path):
            if filename.endswith(".txt"):
                with open(os.path.join(kb_path, filename), 'r', encoding='utf-8') as f:
                    knowledge_text += f"\n--- Source: {filename} ---\n{f.read()}\n"
    return knowledge_text

@app.post('/run')
def run_security_audit(task: Task):
    logs = []
    bandit_results = ""
    semgrep_results = ""
    safety_results = ""

    # 1. Load the Knowledge Base (The "Truth" files)
    knowledge = load_knowledge_base()
    logs.append("Knowledge base loaded.")

    # 2. Static Analysis (Bandit & Semgrep)
    if task.context:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(task.context)
            temp_path = f.name
        try:
            b_proc = subprocess.run(['bandit', '-r', temp_path, '-f', 'txt'], capture_output=True, text=True, timeout=30)
            bandit_results = b_proc.stdout
            logs.append("Bandit scan completed.")

            s_proc = subprocess.run(['semgrep', 'scan', '--config', 'auto', temp_path], capture_output=True, text=True, timeout=30)
            semgrep_results = s_proc.stdout
            logs.append("Semgrep scan completed.")
        except Exception as e:
            logs.append(f"Scanner failure: {str(e)}")
        finally:
            os.unlink(temp_path)

    # 3. Dependency Analysis (Safety)
    try:
        safe_proc = subprocess.run(['safety', 'check'], capture_output=True, text=True, timeout=30)
        safety_results = safe_proc.stdout
        logs.append("Safety scan completed.")
    except Exception as e:
        safety_results = f"Safety Error: {str(e)}"

    # 4. AI REASONING with Knowledge Base integration
    examples = """
    EXAMPLE:
    INPUT: os.system('ls')
    KNOWLEDGE: OWASP #3 Injection - avoid os.system()
    RESULT: REJECTED. High Risk. Use subprocess.run() instead.
    """

    system_prompt = f"""
    You are a Senior Cybersecurity Analyst. 
    
    YOU MUST BASE YOUR AUDIT ON THE FOLLOWING KNOWLEDGE BASE:
    {knowledge}
    
    STYLE GUIDE:
    {examples}
    
    REAL TASK TO AUDIT:
    CODE: {task.context}
    BANDIT: {bandit_results}
    SEMGREP: {semgrep_results}
    SAFETY: {safety_results}
    
    Please provide a report:
    - OVERALL RISK SCORE (Based on OWASP standards)
    - TECHNICAL BREAKDOWN (Reference the OWASP rule number)
    - DEPENDENCY ALERTS
    - REMEDIATION
    - FINAL VERDICT
    """
    
    security_report = llm.invoke(system_prompt)

    return {
        'task_id': task.task_id,
        'status': 'success',
        'result': security_report,
        'summary': 'Knowledge-augmented security audit completed.',
        'next_agent': 'devops',
        'logs': logs
    }