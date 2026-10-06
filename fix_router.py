with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

new_routing = """
    try:
        if is_data_file:
            agents = ["data_science"]
        elif "github.com" in task.description.lower() or "http://" in task.description.lower() or "https://" in task.description.lower():
            agents = ["ai_specialist"]
        elif "data:image" in task.context.lower():
            agents = ["ai_specialist"]
        else:
            # Explicit mentions override LLM
            mentioned = []
            low_desc = task.description.lower()
            
            # Use regex to find order of mentions to keep the pipeline sequential
            agent_keywords = {
                "data_science": ["data science", "data_science", "dummy content", "data agent"],
                "fullstack": ["fullstack", "ui", "web page", "website", "frontend", "react"],
                "security": ["security", "audit"],
                "devops": ["devops", "deploy", "pipeline"]
            }
            
            # Find earliest occurrence of any keyword for each agent
            positions = {}
            for ag, kws in agent_keywords.items():
                pos = -1
                for kw in kws:
                    idx = low_desc.find(kw)
                    if idx != -1 and (pos == -1 or idx < pos):
                        pos = idx
                if pos != -1:
                    positions[ag] = pos
            
            if positions:
                # Sort agents by where they were mentioned in the text
                agents = [ag for ag, _ in sorted(positions.items(), key=lambda x: x[1])]
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
                    agents = ["fullstack"]
                
                if not agents:
                    agents = ["fullstack"]
    except Exception:
        agents = ["fullstack"]
"""

content = re.sub(r'\s*try:\n\s*if is_data_file:.*?(?=    import httpx)', new_routing + '\n', content, flags=re.DOTALL)

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
