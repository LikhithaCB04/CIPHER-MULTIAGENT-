with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

new_run_task = """@app.post('/run')
async def run_task(task: Task):
    task_id = task.task_id or f"task-{uuid.uuid4().hex[:8]}"
    await broadcast({"event": "task_received", "task_id": task_id, "description": task.description})

    # Strict file routing
    is_data_file = False
    if task.context:
        low_ctx = task.context.lower()
        if "csv" in low_ctx or "spreadsheetml" in low_ctx or "excel" in low_ctx or ".xls" in low_ctx:
            is_data_file = True

    try:
        if is_data_file:
            agents = ["data_science"]
        elif "github.com" in task.description.lower() or "http://" in task.description.lower() or "https://" in task.description.lower():
            agents = ["ai_specialist"]
        elif "data:image" in task.context.lower():
            agents = ["ai_specialist"]
        else:
            prompt = f'''
            You are a task router for a multi-agent AI system.
            Given this task: {task.description}
            If the task mentions a git URL, github, or repository, route it to ai_specialist.
            If the task is a general greeting, small talk, or doesn't clearly require data analysis, code generation, security review, or deployment work, default to fullstack.
            Otherwise choose one or more agents from: data_science, fullstack, security, devops.
            Do NOT route to ai_specialist unless a repository is explicitly mentioned.
            CRITICAL: Return ONLY a valid JSON array of strings, no markdown, no backticks, no explanation. Example: ["fullstack"] or ["data_science", "devops"]
            '''
            agents_raw = await asyncio.to_thread(llm.invoke, prompt)
            if "[" in agents_raw and "]" in agents_raw:
                json_str = agents_raw[agents_raw.find("["):agents_raw.rfind("]") + 1]
                agents = json.loads(json_str)
                if not isinstance(agents, list):
                    agents = []
            else:
                agents = choose_agents(task.description)
            
            if not agents:
                agents = choose_agents(task.description)
    except Exception:
        agents = choose_agents(task.description)

    import httpx
    results = []
    has_error = False
    
    # Clarification Check
    if len(task.description) < 100 and "Clarify:" not in task.description:
        clarify_prompt = f"The user asked to '{task.description}'. This is too brief. Ask 3 clarifying questions to gather requirements, and give a blank space or options for them to fill. Format beautifully in Markdown."
        clarification = llm.invoke(clarify_prompt)
        await broadcast({
            "event": "agent_finished",
            "agent": "ai_specialist",
            "task_id": task_id,
            "result_summary": clarification,
            "next_agent": None
        })
        return {
            "status": "success",
            "task_id": task_id,
            "agents_used": ["ai_specialist"],
            "results": [{
                "task_id": task_id,
                "status": "success",
                "summary": "Please clarify your request:",
                "result": clarification
            }]
        }

    async with httpx.AsyncClient(timeout=900.0) as client:
        for index, agent in enumerate(agents):
            await broadcast({"event": "agent_started", "agent": agent, "task_id": task_id})
            service = AGENT_SERVICES.get(agent, AGENT_SERVICES["ai_specialist"])
            url = f"http://{service['host']}:{service['port']}"
            try:
                payload = {
                    "task_id": task_id,
                    "task_type": task.task_type,
                    "description": task.description,
                    "context": task.context,
                    "priority": task.priority,
                }
                
                resp = await client.post(f"{url}/run", json=payload)
                resp.raise_for_status()
                text = resp.text.strip()
                import json
                try:
                    lines = [line for line in text.split('\\n') if line.strip()]
                    if lines:
                        parsed = json.loads(lines[-1])
                        data = parsed.get("output", parsed) if isinstance(parsed, dict) else parsed
                    else:
                        data = {"status": "error", "summary": "Empty output", "result": ""}
                except Exception as ex:
                    data = {"status": "error", "summary": "JSON error", "result": str(ex) + " on text: " + text[:50]}
                
                summary = data.get("summary", "Agent completed successfully.")
                if data.get("status") == "error":
                    has_error = True
                
                next_agent = None
                if index < len(agents) - 1:
                    next_agent = agents[index + 1]
                    
                results.append(data)
                
                await broadcast({
                    "event": "agent_finished",
                    "agent": agent,
                    "task_id": task_id,
                    "result_summary": summary,
                    "next_agent": next_agent,
                })
            except Exception as e:
                has_error = True
                summary = f"Agent {agent} is not reachable at {url} or failed. Error: {str(e)}"
                results.append({"error": summary, "task_id": task_id, "status": "error"})
                await broadcast({
                    "event": "agent_finished",
                    "agent": agent,
                    "task_id": task_id,
                    "result_summary": summary,
                    "next_agent": None,
                })

    return {
        "status": "error" if has_error else "success",
        "task_id": task_id,
        "agents_used": agents,
        "results": results
    }"""

content = re.sub(r"@app\.post\('/run'\)\nasync def run_task\(task: Task\):.*?return \{\n\s*\"status\": \"error\" if has_error else \"success\",\n\s*\"task_id\": task_id,\n\s*\"agents_used\": agents,\n\s*\"results\": results\n\s*\}", new_run_task, content, flags=re.DOTALL)

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
