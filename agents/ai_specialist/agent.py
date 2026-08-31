import os
import subprocess
import tempfile
import textwrap
import urllib.request
from html.parser import HTMLParser
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
from pydantic import BaseModel

app = FastAPI()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

try:
    from langchain_community.llms import Ollama
    llm = Ollama(model="phi3", base_url=OLLAMA_BASE_URL)
except ImportError:
    class MockLLM:
        def invoke(self, prompt): return f"LLM is missing, could not process: {prompt[:50]}"
    llm = MockLLM()

class Task(BaseModel):
    task_id: str
    task_type: str = "orchestrate"
    description: str
    context: str = ""
    priority: Optional[str] = "medium"

class TaskOutput(BaseModel):
    task_id: str
    status: str
    result: str
    summary: str
    next_agent: Optional[str] = None
    logs: List[str]

def run_command(cmd: List[str], cwd: str) -> str:
    try:
        completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
        return completed.stdout.strip() if completed.returncode == 0 else f"exit={completed.returncode}\n{completed.stderr.strip()}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

class SimpleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
    def handle_data(self, data):
        if data.strip():
            self.text_content.append(data.strip())
    def get_text(self):
        return " ".join(self.text_content)

@app.get('/health')
async def health_check(): return {"status": "ok"}

@app.post("/run")
def process_task(data: Task):
    def generator():
        logs = []
        def log(msg):
            logs.append(msg)
            return json.dumps({"type": "agent_thought", "text": msg}) + "\n"
        logs = [f"AI Specialist received task {data.task_id}"]
        
        # Extract URLs from context or description
        text_to_search = f"{data.context} {data.description}"
        repo_url = None
        browser_url = None
        
        for token in text_to_search.split():
            if "github.com" in token or token.endswith(".git"):
                repo_url = token
                # Fix typos like ttps
                if repo_url.startswith("ttps"): repo_url = "h" + repo_url
            elif token.startswith("http"):
                if not repo_url: browser_url = token
    
        if repo_url:
            yield log(f"Detected Git repository: {repo_url}")
            with tempfile.TemporaryDirectory(prefix="repo-operator-") as temp_dir:
                clone_target = os.path.join(temp_dir, "repo")
                clone_result = run_command(["git", "clone", repo_url, clone_target], cwd=temp_dir)
                yield log(f"Clone result: {clone_result[:100]}")
                
                if clone_result.startswith("exit="):
                    yield json.dumps({"type": "task_output", "output": dict(task_id=data.task_id, status="error", result=clone_result, summary="Failed to clone repository.", logs=logs)}) + "\n"
                return
    
                # Load contents of all files to give context to the LLM
                files_list = run_command(["git", "ls-tree", "-r", "HEAD", "--name-only"], cwd=clone_target)
                
                repo_context = ""
                for file_path in files_list.split('\n'):
                    file_path = file_path.strip()
                    if not file_path or "node_modules" in file_path or "package-lock" in file_path or file_path.endswith((".png", ".jpg", ".ico")): 
                        continue
                    full_path = os.path.join(clone_target, file_path)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if len(content) < 20000: # skip huge files
                                repo_context += f"--- {file_path} ---\n{content}\n\n"
                    except Exception:
                        pass
                    if len(repo_context) > 40000: # Truncate massive repositories
                        repo_context += "\n[... truncated for context window limit ...]"
                        break
                
                prompt = f"Task: {data.description}\n\nRepository Contents:\n{repo_context}\n\nIf the user is asking a question about the repository, answer it directly and comprehensively based on the file contents. If the user explicitly asks to modify or write code to a specific file, return ONLY the exact filepath to modify (e.g. 'index.html') and absolutely nothing else."
                response_text = llm.invoke(prompt).strip()
                
                target_file = None
                for line in files_list.split("\n"):
                    clean_line = line.strip()
                    if clean_line and (clean_line == response_text or clean_line == response_text.strip("`'\"")):
                        target_file = clean_line
                        break
                        
                if not target_file:
                    # It's an answer to a question (or it didn't return a valid file)
                    yield json.dumps({"type": "task_output", "output": dict(
                        task_id=data.task_id, status="success",
                        result=response_text,
                        summary=f"Analyzed repository {repo_url}.",
                        logs=logs
                    )}) + "\n"
                return
                    
                # Otherwise, we edit the file
                full_path = os.path.join(clone_target, target_file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        old_content = f.read()
                except Exception:
                    old_content = ""
                    
                prompt_edit = f"Update this code to fulfill the task: {data.description}\n\nCode:\n{old_content}\n\nReturn ONLY the fully updated code, no markdown wrapping."
                new_content = llm.invoke(prompt_edit).strip()
                if new_content.startswith("```"):
                    new_content = "\n".join(new_content.split("\n")[1:-1])
                    
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                    
                run_command(["git", "add", "."], cwd=clone_target)
                run_command(["git", "commit", "-m", f"AI Auto-update: {data.description[:30]}"], cwd=clone_target)
                push_res = run_command(["git", "push"], cwd=clone_target)
                
                yield json.dumps({"type": "task_output", "output": dict(
                    task_id=data.task_id, status="success",
                    result=f"Edited {target_file}.\nPush result: {push_res}",
                    summary=f"Cloned {repo_url}, edited {target_file}, and committed changes.",
                    logs=logs
                )}) + "\n"
                return
                
        elif browser_url:
            yield log(f"Detected Browser URL: {browser_url}")
            try:
                req = urllib.request.Request(browser_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8')
                parser = SimpleTextParser()
                parser.feed(html)
                page_text = parser.get_text()[:4000] # truncate for LLM
                
                answer = llm.invoke(f"Based on this webpage content: {page_text}\n\nTask: {data.description}")
                yield json.dumps({"type": "task_output", "output": dict(task_id=data.task_id, status="success", result=answer, summary=f"Analyzed {browser_url}", logs=logs)}) + "\n"
                return
            except Exception as e:
                yield json.dumps({"type": "task_output", "output": dict(task_id=data.task_id, status="error", result=str(e), summary="Failed to load webpage.", logs=logs)}) + "\n"
                return
    
        # Handle attached images via context
        if "data:image" in data.context:
            yield log("Detected image in context, routing to Llava vision model...")
            import requests
            # Find the base64 string
            start_idx = data.context.find("data:image")
            newline_idx = data.context.find("\n", start_idx)
            if newline_idx == -1: newline_idx = len(data.context)
            b64_data = data.context[start_idx:newline_idx]
            
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
                
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": "llava",
                        "prompt": data.description or "Describe this image.",
                        "images": [b64_data],
                        "stream": False
                    },
                    timeout=180
                )
                if resp.status_code == 200:
                    answer = resp.json().get("response", "")
                    yield json.dumps({"type": "task_output", "output": {"task_id": data.task_id, "status": "success", "result": answer, "summary": "Analyzed uploaded image with Llava.", "logs": logs}}) + "\n"
                    return
                else:
                    err_msg = resp.text
                    if "model 'llava' not found" in err_msg.lower():
                        yield json.dumps({"type": "task_output", "output": {"task_id": data.task_id, "status": "error", "result": "Llava model is not installed. Please run `ollama pull llava` in your terminal to enable image support.", "summary": "Missing Llava vision model.", "logs": logs}}) + "\n"
                        return
                    yield json.dumps({"type": "task_output", "output": {"task_id": data.task_id, "status": "error", "result": f"Ollama API Error: {err_msg}", "summary": "Failed to analyze image.", "logs": logs}}) + "\n"
                    return
            except Exception as e:
                yield json.dumps({"type": "task_output", "output": {"task_id": data.task_id, "status": "error", "result": f"Error contacting Ollama: {str(e)}", "summary": "Vision model failed.", "logs": logs}}) + "\n"
                return
    
        yield json.dumps({"type": "task_output", "output": {
            "task_id": data.task_id, "status": "success",
            "result": "No repository URL or web link was supplied.",
            "summary": "No links provided. Skipping operations.",
            "logs": logs + ["Missing links"]
        }}) + "\n"
        return
    
    return StreamingResponse(generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
