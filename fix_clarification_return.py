with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

new_return = """            return {
                "status": "success",
                "task_id": task_id,
                "agents_used": ["ai_specialist"],
                "results": [{
                    "task_id": task_id,
                    "status": "success",
                    "summary": "Please clarify your request:",
                    "result": clarification
                }]
            }"""

content = content.replace('return {"status": "clarification_needed"}', new_return)

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
