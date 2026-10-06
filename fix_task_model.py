with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('    priority: Optional[str] = "medium"', '    priority: Optional[str] = "medium"\n    history: Optional[list] = None')

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
