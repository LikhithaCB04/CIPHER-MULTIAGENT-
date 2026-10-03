with open("frontend/src/CanvasApp.tsx", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("> SYSTEM_STANDBY", "&gt; SYSTEM_STANDBY")
content = content.replace("> TASK_COMPLETED [OK]", "&gt; TASK_COMPLETED [OK]")

with open("frontend/src/CanvasApp.tsx", "w", encoding="utf-8") as f:
    f.write(content)
