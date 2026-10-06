with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("text.split('\n", "text.split('\\n")

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
