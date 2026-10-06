import os
import re
import urllib.request
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

from groq import Groq
import os
import asyncio

class LLMProxy:
    def __init__(self, model_name="gemini-3.1-pro", max_tokens=2500):
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
    async def astream(self, prompt):
        import asyncio
        response = await asyncio.to_thread(self.invoke, prompt)
        yield response

    def invoke(self, prompt: str) -> str:
        import requests, os
        payload = {"contents": [{"parts":[{"text": prompt}]}]}
        try:
            resp = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Fallback to HF if Gemini quota is exhausted
            if resp.status_code == 429 or "error" in data:
                hf_key = os.environ.get("HF_API_KEY", "")
                hf_res = requests.post("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
                                       headers={"Authorization": f"Bearer {hf_key}"},
                                       json={"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": prompt}]})
                if hf_res.status_code == 200:
                    return hf_res.json()["choices"][0]["message"]["content"]
                
                # If that fails, try via huggingface_hub client if available
                try:
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=hf_key)
                    res = client.chat_completion([{"role": "user", "content": prompt}], model="Qwen/Qwen2.5-72B-Instruct")
                    return res.choices[0].message.content
                except Exception:
                    pass

            return f"Error: {data}"
        except Exception as e:
            return f"Error: {str(e)}"


def parse_markdown_files(md_content):
    import re
    files = {}
    allowlist = {
        "package.json", "index.html", "src/main.jsx", "src/App.jsx", "src/App.css"
    }
    pattern = re.compile(
        r'(?:(?:FILE:|###|\*\*)\s*([a-zA-Z0-9_./-]+)(?:\*\*)?\s*\n)?```[a-zA-Z0-9-]*\n(?:(?://|/\*|<!--)\s*([a-zA-Z0-9_./-]+)\s*\n)?(.*?)(?:\n```|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(md_content):
        filepath = (m.group(1) or m.group(2))
        content = m.group(3)
        if not filepath:
            continue
        filepath = filepath.strip()
        if content:
            content = content.replace("end-of-content", "")
            content = content.replace("END_OF_FILES", "")
            content = content.strip()
        if filepath in allowlist:
            files[filepath] = content
    return files

llm = LLMProxy(model_name=os.environ.get("GROQ_AGENT_MODEL", "openai/gpt-oss-120b"), max_tokens=2500)


class TaskInput(BaseModel):
    task_id: str
    task_type: str = "fullstack"
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


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/run")
async def process_task(data: TaskInput, request: Request):

    async def generator():

        logs = []

        def log(msg):
            logs.append(msg)
            return json.dumps({
                "type": "agent_thought",
                "text": msg
            }) + "\n"

        yield log("Analyzing fullstack requirements...")

        try:
            yield log("Generating files in 2 stages...")
            
            async def generate_chunk(prompt_text):
                content = ""
                async for chunk in llm.astream(prompt_text):
                    content += chunk
                return content.strip()

            base_rules = f"""
Task: {data.description}
Context: {data.context}

You are a code generator. Do not discuss the task.
ABSOLUTE RULE: No external dependencies other than react and react-dom.
ABSOLUTE RULE: Do NOT import any local files other than ./App and ./App.css. Everything else MUST be inline.
ABSOLUTE RULE: Do NOT use fake placeholders or omit implementations.
"""

            yield log("Injecting fixed Vite infrastructure files...")
            c1 = """FILE: package.json
```json
{
  "name": "portfolio",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^4.4.5"
  }
}
```

FILE: index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>React App</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
```

FILE: src/main.jsx
```jsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './App.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```"""

            yield log("Call 1: Generating UI files (App.jsx, App.css)...")
            p2 = base_rules + """
Generate the implementation for the application. You must output the required files in FILE blocks, but you should also INCLUDE detailed setup instructions, explanations, and how to run it, similar to an AI coding assistant.
Output files in this format:
FILE: src/App.jsx
```jsx
...
```
"""
            c2 = await generate_chunk(p2)
            if await request.is_disconnected(): return

            generated_code = c1 + "\n\n" + c2

            yield log("Code generation complete. Parsing files...")

            files = parse_markdown_files(generated_code)

            preview_url = None
            server_error = None

            safe_task_id = re.sub(
                r"[^a-zA-Z0-9_-]",
                "",
                data.task_id
            ) or "default_task"

            os.makedirs("workspace", exist_ok=True)
            with open(f"workspace/{safe_task_id}_raw.txt", "w", encoding="utf-8") as f:
                f.write(generated_code)

            required_files = {"package.json", "index.html", "src/main.jsx", "src/App.jsx", "src/App.css"}

            unsupported_import = None
            found_bad_pattern = None
            found_semantic_error = None
            if len(files) >= 3:
                for filepath in ["src/App.jsx", "src/main.jsx"]:
                    content = files.get(filepath, "")
                    import_pattern = re.compile(r"^\s*import\s+(?:.*?\s+from\s+)?['\"](.*?)['\"]", re.MULTILINE)
                    for match in import_pattern.finditer(content):
                        dep = match.group(1)
                        if dep not in ["react", "react-dom", "react-dom/client", "./App", "./App.css"]:
                            unsupported_import = dep
                            break
                    if unsupported_import:
                        break

                bad_patterns = [
                    "/path/to/", "project1.jpg", "project2.jpg", "project3.jpg",
                    "end-of-content", "END_OF_FILES", "##", "# ",
                    "TODO", "FIXME", "coming soon", "// placeholder", "/* placeholder", "{/* placeholder", "YOUR_CODE_HERE",
                    "implementation omitted", "code omitted",
                    "{/* Navbar content */}", "{/* Project cards */}",
                    "{/* Add content here */}", "{/* More sections */}"
                ]
                found_bad_pattern = None
                if not unsupported_import:
                    for filepath in ["src/App.jsx", "src/App.css", "index.html"]:
                        content = files.get(filepath, "")
                        
                        if filepath == "src/App.css" and "@import" in content:
                            found_bad_pattern = "@import"
                            break
                            
                        if filepath == "index.html" and "%PUBLIC_URL%" in content:
                            found_bad_pattern = "%PUBLIC_URL%"
                            break
                            
                        for pattern in bad_patterns:
                            if pattern in content:
                                found_bad_pattern = pattern
                                break
                        if found_bad_pattern:
                            break

                semantic_bad_patterns = [
                    "your complete", "define your", "components here",
                    "should be developed", "due to brevity",
                    "additional functionalities", "here is", "note:"
                ]
                if not unsupported_import and not found_bad_pattern:
                    app_jsx_content = files.get("src/App.jsx", "")
                    content_lower = app_jsx_content.lower()
                    for pattern in semantic_bad_patterns:
                        if pattern in content_lower:
                            found_semantic_error = f"Template text detected: '{pattern}'"
                            break
                            
                    

            if unsupported_import:
                server_error = f"Generated code uses an unsupported dependency: {unsupported_import}"
                yield log(server_error)
            elif found_bad_pattern:
                server_error = f"Generated code contains forbidden pattern: {found_bad_pattern}"
                yield log(server_error)
            elif found_semantic_error:
                server_error = f"Semantic validation failed: {found_semantic_error}"
                yield log(server_error)
            elif len(files) >= 3:

                workspace_dir = os.path.abspath(
                    os.path.join(
                        os.getcwd(),
                        "workspace",
                        safe_task_id
                    )
                )

                os.makedirs(
                    workspace_dir,
                    exist_ok=True
                )

                yield log(
                    f"Writing {len(files)} files to workspace/{safe_task_id}..."
                )

                for filepath, content in files.items():

                    clean_path = filepath.lstrip("/")

                    target_path = os.path.abspath(
                        os.path.join(
                            workspace_dir,
                            clean_path
                        )
                    )

                    if target_path.startswith(workspace_dir):

                        os.makedirs(
                            os.path.dirname(target_path),
                            exist_ok=True
                        )

                        with open(
                            target_path,
                            "w",
                            encoding="utf-8"
                        ) as f:
                            f.write(content)

                if "package.json" in files:

                    yield log(
                        "React/Vite project detected. Validating project structure..."
                    )

                    pkg_path = os.path.join(
                        workspace_dir,
                        "package.json"
                    )

                    with open(
                        pkg_path,
                        "r",
                        encoding="utf-8"
                    ) as f:
                        pkg_content = f.read()

                    # Sanitize obvious LLM artifacts.
                    pkg_clean = re.sub(
                        r",\s*}",
                        "}",
                        pkg_content
                    )

                    pkg_clean = re.sub(
                        r",\s*]",
                        "]",
                        pkg_clean
                    )

                    pkg_clean = re.sub(
                        r"^\s*in\s*$",
                        "",
                        pkg_clean,
                        flags=re.MULTILINE
                    )

                    try:
                        pkg_json = json.loads(pkg_clean)
                    except json.JSONDecodeError as e:
                        yield log(f"Malformed package.json from LLM: {str(e)}. Repairing with default structure...")
                        pkg_json = {}

                    # -------------------------------------------------
                    # DETERMINISTIC REACT + VITE RUNTIME
                    # -------------------------------------------------
                    # Do NOT trust arbitrary dependencies generated
                    # by the LLM. This prevents hallucinated packages
                    # such as @unplugin/unplugin-react.
                    pkg_json["dependencies"] = {
                        "react": "^18.2.0",
                        "react-dom": "^18.2.0"
                    }

                    pkg_json["devDependencies"] = {
                        "@vitejs/plugin-react": "^4.0.0",
                        "vite": "^5.0.0"
                    }

                    pkg_json["scripts"] = {
                        "dev": "vite",
                        "build": "vite build"
                    }

                    with open(
                        pkg_path,
                        "w",
                        encoding="utf-8"
                    ) as f:
                        json.dump(
                            pkg_json,
                            f,
                            indent=2
                        )

                    yield log(
                        "package.json parsed successfully."
                    )

                    if not server_error:
                        yield log(
                            "Validating React/Vite runtime configuration..."
                        )

                        # -------------------------------------------------
                        # Deterministic Vite configuration
                        # -------------------------------------------------
                        vite_conf_path = os.path.join(
                            workspace_dir,
                            "vite.config.js"
                        )

                        with open(
                            vite_conf_path,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            f.write(
                                "import { defineConfig } from 'vite';\n"
                                "import react from '@vitejs/plugin-react';\n\n"
                                "export default defineConfig({\n"
                                "  plugins: [react()],\n"
                                "});\n"
                            )

                        # Remove conflicting Vite TypeScript config.
                        for alt in ["vite.config.ts"]:

                            alt_path = os.path.join(
                                workspace_dir,
                                alt
                            )

                            if os.path.exists(alt_path):
                                os.remove(alt_path)

                        yield log(
                            "Vite configuration validated."
                        )

                        # -------------------------------------------------
                        # Deterministic React entry point
                        # -------------------------------------------------
                        main_jsx_path = os.path.join(
                            workspace_dir,
                            "src",
                            "main.jsx"
                        )

                        os.makedirs(
                            os.path.dirname(main_jsx_path),
                            exist_ok=True
                        )

                        with open(
                            main_jsx_path,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            f.write(
                                "import React from 'react';\n"
                                "import ReactDOM from 'react-dom/client';\n"
                                "import App from './App.jsx';\n\n"
                                "ReactDOM.createRoot("
                                "document.getElementById('root')"
                                ").render(\n"
                                "  <React.StrictMode>\n"
                                "    <App />\n"
                                "  </React.StrictMode>,\n"
                                ");\n"
                            )

                        # Remove conflicting entry points.
                        for alt in [
                            "main.js",
                            "index.js",
                            "index.jsx",
                            "main.tsx"
                        ]:

                            alt_path = os.path.join(
                                workspace_dir,
                                "src",
                                alt
                            )

                            if (
                                os.path.exists(alt_path)
                                and alt_path != main_jsx_path
                            ):
                                os.remove(alt_path)

                        yield log(
                            "React entry point validated."
                        )

                        # -------------------------------------------------
                        # Ensure index.html is valid
                        # -------------------------------------------------
                        index_path = os.path.join(
                            workspace_dir,
                            "index.html"
                        )

                        if os.path.exists(index_path):

                            with open(
                                index_path,
                                "r",
                                encoding="utf-8"
                            ) as f:
                                idx_content = f.read()

                            if 'id="root"' not in idx_content:

                                idx_content = idx_content.replace(
                                    "<body>",
                                    '<body>\n'
                                    '    <div id="root"></div>'
                                )

                            if (
                                'src="/src/main.jsx"' not in idx_content
                                and "src='/src/main.jsx'" not in idx_content
                            ):

                                idx_content = idx_content.replace(
                                    "</body>",
                                    '    <script type="module" '
                                    'src="/src/main.jsx"></script>\n'
                                    "  </body>"
                                )

                            # Strip any other random scripts LLM might hallucinate
                            idx_content = re.sub(
                                r'<script(?![^>]*src=[\'"]/src/main\.jsx[\'"]).*?</script>',
                                "",
                                idx_content,
                                flags=re.DOTALL | re.IGNORECASE
                            )

                            idx_content = idx_content.replace("%PUBLIC_URL%/", "")
                            idx_content = idx_content.replace("%PUBLIC_URL%", "")

                            with open(
                                index_path,
                                "w",
                                encoding="utf-8"
                            ) as f:
                                f.write(idx_content)

                        # -------------------------------------------------
                        # Sanitize framework imports
                        # -------------------------------------------------
                        for root_dir, _, filenames in os.walk(
                            workspace_dir
                        ):

                            for fname in filenames:

                                if fname.endswith(
                                    (".js", ".jsx", ".ts", ".tsx")
                                ) and fname not in [
                                    "vite.config.js",
                                    "main.jsx"
                                ]:

                                    fpath = os.path.join(
                                        root_dir,
                                        fname
                                    )

                                    with open(
                                        fpath,
                                        "r",
                                        encoding="utf-8"
                                    ) as f:
                                        content = f.read()

                                    orig_content = content

                                    content = re.sub(
                                        r'import\s+.*?from\s+[\'"]next/head[\'"];?',
                                        "",
                                        content
                                    )

                                    content = re.sub(
                                        r"<Head>.*?</Head>",
                                        "",
                                        content,
                                        flags=re.DOTALL
                                    )

                                    content = content.replace(
                                        "@vitejs/plugin-react-ssr",
                                        "@vitejs/plugin-react"
                                    )

                                    if content != orig_content:

                                        with open(
                                            fpath,
                                            "w",
                                            encoding="utf-8"
                                        ) as f:
                                            f.write(content)

                        # -------------------------------------------------
                        # Install dependencies
                        # -------------------------------------------------
                        yield log(
                            "Installing dependencies (this may take a minute)..."
                        )

                        npm_cmd = (
                            "npm.cmd"
                            if os.name == "nt"
                            else "npm"
                        )

                        proc = await asyncio.create_subprocess_exec(
                            npm_cmd,
                            "install",
                            cwd=workspace_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )

                        stdout, stderr = await proc.communicate()

                        if proc.returncode != 0:

                            server_error = (
                                "Dependency installation failed:\n"
                                + stderr.decode(
                                    "utf-8",
                                    errors="ignore"
                                )
                            )

                            yield log(server_error)

                        else:

                            yield log("Validating generated React code via build...")
                            build_proc = await asyncio.create_subprocess_exec(
                                npm_cmd, "run", "build",
                                cwd=workspace_dir,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            b_stdout, b_stderr = await build_proc.communicate()

                            if build_proc.returncode != 0:
                                b_err_str = b_stderr.decode("utf-8", errors="ignore") + "\n" + b_stdout.decode("utf-8", errors="ignore")
                                yield log("Build failed. Attempting LLM repair...")

                                repaired = False
                                app_jsx_path = os.path.join(workspace_dir, "src", "App.jsx")
                                app_css_path = os.path.join(workspace_dir, "src", "App.css")

                                if os.path.exists(app_jsx_path) and os.path.exists(app_css_path):
                                    with open(app_jsx_path, "r", encoding="utf-8") as f:
                                        app_content = f.read()
                                    with open(app_css_path, "r", encoding="utf-8") as f:
                                        css_content = f.read()
                                        
                                    yield log("Asking LLM to repair the build error...")
                                    repair_prompt = f"""Repair the existing generated React code. Do not redesign it. Do not remove requested functionality. Fix only syntax, JSX structure, imports, undefined variables, and other build errors shown by the compiler. Return the COMPLETE corrected file. Do not include explanations, markdown outside the FILE block, prompt text, duplicated code, `svgsvg`, TODOs, placeholders, or unrelated task content.

Vite Build Error:
{b_err_str}

Current App.jsx:
```jsx
{app_content}
```

Current App.css:
```css
{css_content}
```

Output EXACTLY these two file blocks in the standard format:
FILE: src/App.jsx
```jsx
...
```

FILE: src/App.css
```css
...
```
"""
                                    repaired_code = await generate_chunk(repair_prompt)
                                    repaired_files = parse_markdown_files(repaired_code)
                                    
                                    if "src/App.jsx" in repaired_files:
                                        with open(app_jsx_path, "w", encoding="utf-8") as f:
                                            f.write(repaired_files["src/App.jsx"])
                                        repaired = True
                                    if "src/App.css" in repaired_files:
                                        with open(app_css_path, "w", encoding="utf-8") as f:
                                            f.write(repaired_files["src/App.css"])

                                if repaired:
                                    yield log("Repair applied. Retrying build...")
                                    build_proc_2 = await asyncio.create_subprocess_exec(
                                        npm_cmd, "run", "build",
                                        cwd=workspace_dir,
                                        stdout=asyncio.subprocess.PIPE,
                                        stderr=asyncio.subprocess.PIPE
                                    )
                                    b2_stdout, b2_stderr = await build_proc_2.communicate()
                                    if build_proc_2.returncode != 0:
                                        server_error = "React build failed after repair:\n" + b2_stderr.decode("utf-8", errors="ignore") + "\n" + b2_stdout.decode("utf-8", errors="ignore")
                                        yield log(server_error)
                                else:
                                    server_error = "React build failed:\n" + b_err_str
                                    yield log(server_error)

                            if not server_error:
                                yield log(
                                    "Build validated. "
                                    "Starting Vite server on port 5174..."
                                )

                                # -------------------------------------------------
                                # Start generated website
                                # -------------------------------------------------
                                subprocess.Popen(
                                    [
                                        npm_cmd,
                                        "run",
                                        "dev",
                                        "--",
                                        "--host",
                                        "127.0.0.1",
                                        "--port",
                                        "5174",
                                        "--strictPort"
                                    ],
                                    cwd=workspace_dir,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )

                                yield log(
                                    "Waiting for server to become ready "
                                    "(up to 15s)..."
                                )

                                ready = False

                                preview_url = (
                                    "http://127.0.0.1:5174"
                                )

                                for _ in range(15):

                                    try:

                                        urllib.request.urlopen(
                                            preview_url,
                                            timeout=1
                                        )

                                        ready = True
                                        break

                                    except Exception:

                                        await asyncio.sleep(1)

                                if not ready:

                                    server_error = (
                                        "Server failed to start or bind "
                                        "to port 5174 within 15 seconds."
                                    )

                                    yield log(server_error)

                                    preview_url = None

            else:
                server_error = "The LLM failed to generate at least 3 required files."
                yield log(server_error)

            # -------------------------------------------------------------
            # Prepare final response
            # -------------------------------------------------------------
            if server_error:

                final_status = "error"
                final_summary = server_error

            else:

                final_status = "success"

                missing_files = required_files - set(files.keys())
                note = f" Note: missing {', '.join(missing_files)}" if missing_files else ""
                final_summary = f"Successfully generated fullstack application code.{note}"

                if preview_url:

                    final_summary += (
                        "\n\nApplication generated successfully."
                        f"\nProject location: "
                        f"`workspace/{safe_task_id}/`"
                        f"\nLive preview: {preview_url}"
                    )

            yield json.dumps({
                "type": "task_output",
                "output": {
                    "task_id": data.task_id,
                    "status": final_status,
                    "result": generated_code,
                    "summary": final_summary,
                    "next_agent": None,
                    "logs": logs
                }
            }) + "\n"

        except asyncio.CancelledError:
            # Client abruptly terminated connection.
            raise

        except Exception as e:

            yield log(
                f"Error during LLM generation: {str(e)}"
            )

            yield json.dumps({
                "type": "task_output",
                "output": {
                    "task_id": data.task_id,
                    "status": "error",
                    "result": (
                        f"LLM Generation failed: {str(e)}"
                    )
                }
            }) + "\n"

        return

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson"
    )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002
    )
