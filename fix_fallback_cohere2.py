import re

files = [
    "agents/ai_specialist/agent.py",
    "agents/ayeesha_fullstack/ayeesha_agent.py",
    "orchestrator/orchestrator.py"
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace('"command-r-plus"', '"command-a-03-2025"')
    
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)
