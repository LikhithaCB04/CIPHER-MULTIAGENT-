with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

import re
replacement = """    import httpx
    results = []
    
    has_error = False
    
    # Clarification Check
    if len(task.description) < 100 and "Clarify:" not in task.description:
        clarify_prompt = f"The user asked to '{task.description}'. This is too brief. Ask 3 clarifying questions to gather requirements, and give a blank space or options for them to fill. Format beautifully in Markdown."
        clarification = router_llm.invoke(clarify_prompt)
        await broadcast({
            "event": "agent_finished",
            "agent": "ai_specialist",
            "task_id": task_id,
            "result_summary": clarification,
            "next_agent": None
        })
        return {"status": "clarification_needed"}

    async with httpx.AsyncClient(timeout=900.0) as client:
"""
content = re.sub(r'    import httpx\s*results = \[\]\s*has_error = False\s*# Clarification Check.*?async with httpx\.AsyncClient\(timeout=900\.0\) as client:', replacement, content, flags=re.DOTALL)

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
