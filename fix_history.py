with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Add history to the clarification check
replacement = """    if len(task.description) < 100 and "Clarify:" not in task.description and not task.history:"""
content = re.sub(r'    if len\(task\.description\) < 100 and "Clarify:" not in task\.description:', replacement, content)

# Pass history in payload
content = content.replace('"priority": task.priority,', '"priority": task.priority,\n                    "history": task.history,')

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
