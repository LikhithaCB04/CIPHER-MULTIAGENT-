with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if "has_error = False" in lines[i] and not "        has_error = False" in lines[i]:
        pass
    if lines[i].startswith("        has_error = False"):
        lines[i] = "    has_error = False\n"
    elif lines[i].startswith("        # Clarification Check"):
        lines[i] = "    # Clarification Check\n"
    elif lines[i].startswith("        if len(task.description)"):
        lines[i] = "    if len(task.description) < 100 and \"Clarify:\" not in task.description:\n"
    elif lines[i].startswith("            clarify_prompt"):
        lines[i] = "        clarify_prompt = f\"The user asked to '{task.description}'. This is too brief. Ask 3 clarifying questions to gather requirements, and give a blank space or options for them to fill. Format beautifully in Markdown.\"\n"
    elif lines[i].startswith("            clarification = router_llm.invoke"):
        lines[i] = "        clarification = router_llm.invoke(clarify_prompt)\n"
    elif lines[i].startswith("            await broadcast({"):
        lines[i] = "        await broadcast({\n"
    elif lines[i].startswith("                \"event\": \"agent_finished\","):
        lines[i] = "            \"event\": \"agent_finished\",\n"
    elif lines[i].startswith("                \"agent\": \"ai_specialist\","):
        lines[i] = "            \"agent\": \"ai_specialist\",\n"
    elif lines[i].startswith("                \"task_id\": task_id,"):
        lines[i] = "            \"task_id\": task_id,\n"
    elif lines[i].startswith("                \"result_summary\": clarification,"):
        lines[i] = "            \"result_summary\": clarification,\n"
    elif lines[i].startswith("                \"next_agent\": None"):
        lines[i] = "            \"next_agent\": None\n"
    elif lines[i].startswith("            })"):
        lines[i] = "        })\n"
    elif lines[i].startswith("            return {\"status\": \"clarification_needed\"}"):
        lines[i] = "        return {\"status\": \"clarification_needed\"}\n"
    elif lines[i].startswith("        async with httpx.AsyncClient"):
        lines[i] = "    async with httpx.AsyncClient(timeout=900.0) as client:\n"

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
