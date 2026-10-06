with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("router_llm.invoke", "llm.invoke")

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
