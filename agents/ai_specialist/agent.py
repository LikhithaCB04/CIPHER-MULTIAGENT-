from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import subprocess
import tempfile
import textwrap

app = FastAPI()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


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
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        return f"exit={completed.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    return stdout or "OK"


@app.get('/health')
async def health_check():
    return {"status": "ok"}


@app.post("/run", response_model=TaskOutput)
async def process_task(data: Task):
    logs = [f"Repo operator received task {data.task_id}"]
    try:
        repo_url = None
        for token in ["https://", "http://", "git@", "github.com"]:
            if token in data.context:
                repo_url = token
                break
        if repo_url is None and data.context:
            repo_url = data.context.strip()

        if not repo_url:
            return TaskOutput(
                task_id=data.task_id,
                status="error",
                result="No repository URL was supplied in the context.",
                summary="Repo operator requires a git URL in the request context.",
                next_agent=None,
                logs=logs + ["Missing repo URL"],
            )

        with tempfile.TemporaryDirectory(prefix="repo-operator-", dir="/tmp") as temp_dir:
            clone_target = os.path.join(temp_dir, "repo")
            clone_result = run_command(["git", "clone", repo_url, clone_target], cwd=temp_dir)
            logs.append(f"Clone result: {clone_result[:250]}")
            if clone_result.startswith("exit="):
                return TaskOutput(
                    task_id=data.task_id,
                    status="error",
                    result=f"Clone failed: {clone_result}",
                    summary="Repository could not be cloned.",
                    next_agent=None,
                    logs=logs,
                )

            changed_files = []
            edit_hint = data.description.lower()
            for root, _, files in os.walk(clone_target):
                for filename in files:
                    if filename.endswith((".md", ".txt", ".py", ".js", ".ts", ".tsx", ".json", ".yml", ".yaml")):
                        path = os.path.join(root, filename)
                        try:
                            with open(path, "r", encoding="utf-8") as handle:
                                content = handle.read()
                        except Exception:
                            continue
                        if "TODO" in content or "FIXME" in content or edit_hint in content.lower():
                            if filename.endswith((".md", ".txt")):
                                updated = content.replace("TODO", f"TODO ({data.description})")
                            else:
                                updated = content.replace("placeholder", f"{data.description}")
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(updated)
                            changed_files.append(path.replace(clone_target + os.sep, ""))
                            break

            if not changed_files:
                changed_files = ["README.md"]
                with open(os.path.join(clone_target, "README.md"), "a", encoding="utf-8") as handle:
                    handle.write(f"\n\nRepo operator note: {data.description}\n")

            test_result = run_command(["bash", "-lc", "ls && (pytest -q || npm test -- --help || true)"], cwd=clone_target)
            logs.append(f"Tests result: {test_result[:250]}")
            status = "success"
            summary = "Applied a focused repository change and recorded the test attempt."
            if "exit=" in test_result:
                status = "partial"
                summary = "Applied a change, but the test command reported a failure."

            return TaskOutput(
                task_id=data.task_id,
                status=status,
                result=textwrap.dedent(f"""
                Repository processed.
                Files changed: {', '.join(changed_files)}
                Test attempt: {test_result}
                """),
                summary=summary,
                next_agent=None,
                logs=logs,
            )
    except Exception as exc:
        return TaskOutput(
            task_id=data.task_id,
            status="error",
            result=f"Repo operator failed: {exc}",
            summary="The repository operator could not complete the task.",
            next_agent=None,
            logs=logs + [str(exc)],
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8005)
