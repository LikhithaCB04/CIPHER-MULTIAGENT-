from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
import subprocess
import tempfile
import os

# 1. Setup the API and AI Model
app = FastAPI()
llm = Ollama(model="mistral") 

# 2. Define the API Contract (Exactly as per contract.json)
class Task(BaseModel):
    task_id: str
    description: str
    context: str  # This is the code that will be sent for auditing

@app.post('/run')
def run_security_audit(task: Task):
    logs = []
    bandit_results = ""

    # --- PHASE 1: STATIC ANALYSIS (The Truth) ---
    # We use Bandit to find real bugs. We save the code to a temp file first.
    if task.context:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(task.context)
            temp_path = f.name

        try:
            # Run Bandit: -r (recursive), -f txt (text output)
            process = subprocess.run(['bandit', '-r', temp_path, '-f', 'txt'], 
                                    capture_output=True, text=True, timeout=30)
            bandit_results = process.stdout
            logs.append("Bandit static analysis completed.")
        except Exception as e:
            bandit_results = f"Scanner Error: {str(e)}"
            logs.append(f"Scanner failed: {str(e)}")
        finally:
            os.unlink(temp_path) # Clean up the temporary file
    else:
        logs.append("No code provided for audit.")

    # --- PHASE 2: AI REASONING (The Expert) ---
    # We feed the raw code AND the bandit results to Mistral to get a human report
    system_prompt = f"""
    You are a Senior Cybersecurity Analyst with OWASP expertise. 
    Your job is to audit the following code and the results of a Bandit scan.

    CODE TO AUDIT:
    {task.context}

    BANDIT SCAN RESULTS:
    {bandit_results}

    Please provide a professional Security Audit Report:
    1. RISK LEVEL: (Critical/High/Medium/Low)
    2. VULNERABILITIES: (List exactly what is wrong)
    3. FIXES: (Provide the corrected code snippets)
    4. FINAL VERDICT: (Approved / Rejected)
    """
    
    security_report = llm.invoke(system_prompt)

    # --- PHASE 3: RETURN RESULT (Strictly following contract.json) ---
    return {
        'task_id': task.task_id,
        'status': 'success',
        'result': security_report,
        'summary': 'Security audit complete. Results generated using Bandit and Mistral.',
        'next_agent': 'devops',  # This tells the orchestrator to send it to Likitha next
        'logs': logs
    }

# Command to run: uvicorn agents.mahima_security.mahima_agent:app --port 8003 --reload