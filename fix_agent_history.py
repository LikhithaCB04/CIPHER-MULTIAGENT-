import os

files = [
    "agents/ai_specialist/agent.py",
    "agents/ayeesha_fullstack/ayeesha_agent.py",
    "agents/deepthi_data/deepthi_agent.py",
    "agents/likitha_devops/agent.py",
    "agents/mahima_security/mahima_agent.py"
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Add history field to Task payload
    content = content.replace('    priority: Optional[str] = "medium"', '    priority: Optional[str] = "medium"\n    history: Optional[list] = None')
    
    # Append history to prompt
    # Find the variable definition for prompt. Usually it's p1 = f"..." or p = f"..."
    # We will just inject it into the router invocation. But for the agents, we need to find where they construct the prompt.
    # It's easier to just find `description = payload.description` or `task.description` and inject it there if applicable.
    
    if "p1 = f\"\"\"" in content:
        # Fullstack agent
        replacement = """
    hist_text = ""
    if task.history:
        for msg in task.history:
            role = msg.get("role", "user")
            msg_content = msg.get("content", "")
            hist_text += f"{role}: {msg_content}\\n"
    
    p1 = f\"\"\"
    Conversation History:
    {hist_text}
    Current Request:
"""
        content = content.replace('    p1 = f"""', replacement)
        
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)

