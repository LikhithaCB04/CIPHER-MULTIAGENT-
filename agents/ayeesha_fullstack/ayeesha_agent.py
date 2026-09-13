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


def parse_markdown_files(md_content):
    files = {}
    allowlist = {
        "package.json", "index.html", "src/main.jsx", "src/App.jsx", "src/App.css"
    }

    # Match code blocks. We'll look for filenames immediately preceding or inside the block.
    # Pattern looks for optional FILE: or ### followed by filename, then ```
    # Or ``` followed by // filename
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

        # Sanitize LLM artifacts
        if content:
            content = content.replace("end-of-content", "")
            content = content.replace("END_OF_FILES", "")
            content = content.strip()

        if filepath in allowlist:
            files[filepath] = content

    # Fallback: if we didn't get all files, try a simpler split
    if not allowlist.issubset(files.keys()):
        # Just grab all code blocks and try to guess them based on content or order
        blocks = re.findall(r'```[a-zA-Z0-9-]*\n(.*?)(?:\n```|\Z)', md_content, re.DOTALL)

        # If exactly 5 blocks, assume standard order just in case
        if len(blocks) == 5:
            files["package.json"] = blocks[0].strip()
            files["index.html"] = blocks[1].strip()
            files["src/main.jsx"] = blocks[2].strip()
            files["src/App.jsx"] = blocks[3].strip()
            files["src/App.css"] = blocks[4].strip()
        else:
            for block in blocks:
                if '"name"' in block and '"scripts"' in block:
                    files["package.json"] = block.strip()
                elif '<html' in block or '<div id="root">' in block:
                    files["index.html"] = block.strip()
                elif 'createRoot' in block or 'ReactDOM.render' in block or 'ReactDOM.hydrate' in block or 'document.getElementById(\'root\')' in block:
                    if '<App' in block and not 'export default' in block:
                        files["src/main.jsx"] = block.strip()
                    else:
                        files["src/App.jsx"] = block.strip()
                elif 'export default' in block or 'function App' in block or 'const App' in block:
                    files["src/App.jsx"] = block.strip()
                elif 'display:' in block or 'margin:' in block or 'padding:' in block or '--primary-color' in block or '@import' in block:
                    files["src/App.css"] = block.strip()

    # Final sanitization
    for k in files:
        files[k] = files[k].replace("end-of-content", "").replace("END_OF_FILES", "")

    return {k: v for k, v in files.items() if k in allowlist}


try:
    from langchain_ollama import OllamaLLM
except ImportError:
    class OllamaLLM:
        def __init__(self, model: str, base_url: str = "http://localhost:11434"):
            self.model = model
            self.base_url = base_url

        def invoke(self, prompt: str) -> str:
            return f"[mock-fullstack] {self.model} responded to: {prompt[:120]}"

        async def astream(self, prompt: str):
            yield self.invoke(prompt)


app = FastAPI()

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

llm = OllamaLLM(model="phi3", base_url=OLLAMA_BASE_URL, num_predict=800)


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

        prompt = f'''
Task: {data.description}
Context: {data.context}

You are a code generator. Do not discuss the task. Do not explain limitations. Do not argue with the requirements. Generate the requested files immediately.

REQUIREMENTS:
- ONLY App.jsx and App.css need meaningful creative generation.
- Keep App.jsx SHORT (60-80 lines maximum).
- Keep App.css SHORT (100-130 lines maximum).
- App.jsx must include: navbar, hero, skills, exactly 3 project cards, contact.
- DO NOT use external images, external libraries, placeholder image paths, SVG.
- ALL COMPONENTS MUST BE IN src/App.jsx. Start App.jsx exactly with `import React from 'react'; import './App.css';` AND DO NOT ADD ANY OTHER IMPORTS.
- For project cards, use text/gradient/card styling instead of <img> elements. DO NOT use <img /> tags.
- Use simple hardcoded arrays only if necessary.
- Visual design using CSS only: dark background, accent colors using CSS gradients, glass-like cards using rgba, rounded corners, subtle shadows, hover transform, responsive grid, responsive navbar, clear typography, consistent spacing.
- Complete export default App.

YOU MUST OUTPUT ALL 5 FILES EXACTLY AS SHOWN BELOW, REPLACING "// your code here" WITH THE ACTUAL CODE. DO NOT OMIT ANY FILES.

FILE: package.json
```json
{{
  "name": "portfolio",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {{"dev":"vite","build":"vite build"}},
  "dependencies": {{"react":"^18.2.0","react-dom":"^18.2.0"}},
  "devDependencies": {{"vite":"^5.0.0","@vitejs/plugin-react":"^4.0.0"}}
}}
```

FILE: index.html
```html
<!doctype html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI/ML Portfolio</title>
</head>
<body>
<div id="root"></div>
<script type="module" src="/src/main.jsx"></script>
</body>
</html>
```

FILE: src/main.jsx
```jsx
import React from 'react';
import {{createRoot}} from 'react-dom/client';
import App from './App';
import './App.css';
createRoot(document.getElementById('root')).render(<App />);
```

FILE: src/App.jsx
```jsx
import React from 'react';
import './App.css';
// your code here
```

FILE: src/App.css
```css
/* your code here */
```

END_OF_FILES
'''

        try:

            yield log("Generating code with LLM...")

            generated_code = ""

            async for chunk in llm.astream(prompt):

                if await request.is_disconnected():
                    return

                generated_code += chunk

            yield log("Code generation complete. Parsing files...")

            files = parse_markdown_files(generated_code)

            preview_url = None
            server_error = None

            safe_task_id = re.sub(
                r"[^a-zA-Z0-9_-]",
                "",
                data.task_id
            ) or "default_task"

            with open(f"workspace/{safe_task_id}_raw.txt", "w", encoding="utf-8") as f:
                f.write(generated_code)

            required_files = {"package.json", "index.html", "src/main.jsx", "src/App.jsx", "src/App.css"}

            unsupported_import = None
            found_bad_pattern = None
            if required_files.issubset(files.keys()):
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
                    "end-of-content", "END_OF_FILES", "##", "# "
                ]
                found_bad_pattern = None
                if not unsupported_import:
                    for filepath in ["src/App.jsx", "src/App.css"]:
                        content = files.get(filepath, "")
                        for pattern in bad_patterns:
                            if pattern in content:
                                found_bad_pattern = pattern
                                break
                        if found_bad_pattern:
                            break

            if unsupported_import:
                server_error = f"Generated code uses an unsupported dependency: {unsupported_import}"
                yield log(server_error)
            elif found_bad_pattern:
                server_error = f"Generated code contains forbidden pattern: {found_bad_pattern}"
                yield log(server_error)
            elif required_files.issubset(files.keys()):

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
                                yield log("Build failed. Attempting deterministic repair...")

                                repaired = False
                                app_jsx_path = os.path.join(workspace_dir, "src", "App.jsx")

                                if os.path.exists(app_jsx_path):
                                    with open(app_jsx_path, "r", encoding="utf-8") as f:
                                        app_content = f.read()

                                    orig_content = app_content

                                    # Fix missing export default
                                    if "export default" not in app_content:
                                        app_content += "\n\nexport default App;\n"

                                    # Fix missing React import if used without import
                                    if "React is not defined" in b_err_str and "import React" not in app_content:
                                        app_content = "import React from 'react';\n" + app_content

                                    if app_content != orig_content:
                                        with open(app_jsx_path, "w", encoding="utf-8") as f:
                                            f.write(app_content)
                                        repaired = True

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
                server_error = "The LLM failed to generate the exactly required 5 files."
                yield log(server_error)

            # -------------------------------------------------------------
            # Prepare final response
            # -------------------------------------------------------------
            if server_error:

                final_status = "error"
                final_summary = server_error

            else:

                final_status = "success"

                final_summary = (
                    "Successfully generated fullstack application code."
                )

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
