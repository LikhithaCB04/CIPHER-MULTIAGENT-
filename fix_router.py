with open("orchestrator/orchestrator.py", "r", encoding="utf-8") as f:
    content = f.read()

old_prompt = """      prompt = f'''
      You are a task router for a multi-agent AI system.
      Given this task: {task.description}
      If the task mentions a git URL, github, or repository, route it to ai_specialist.
      If the task is a general greeting, small talk, or doesn't clearly require data analysis, code generation, security review, or deployment work, default to fullstack.
      Otherwise choose one or more agents from: data_science, fullstack, security, devops.
      Do NOT route to ai_specialist unless a repository is explicitly mentioned.
      CRITICAL: Return ONLY a valid JSON array of strings, no markdown, no backticks, no explanation. Example: ["fullstack"] or ["data_science", "devops"]
      '''"""

new_prompt = """      has_files = "Yes" if task.context and len(task.context) > 10 else "No"
      prompt = f'''
      You are a task router for a multi-agent AI system.
      Given this task: {task.description}
      Files attached: {has_files}
      
      If the task mentions a git URL, github, or repository, route it to ai_specialist.
      If the task involves cleaning data, data analysis, spreadsheets, or if files are attached and the task says "clean this" or similar, route to data_science.
      If the task is a general greeting, small talk, or doesn't clearly require data analysis, code generation, security review, or deployment work, default to fullstack.
      Otherwise choose one or more agents from: data_science, fullstack, security, devops.
      Do NOT route to ai_specialist unless a repository is explicitly mentioned.
      CRITICAL: Return ONLY a valid JSON array of strings, no markdown, no backticks, no explanation. Example: ["fullstack"] or ["data_science"]
      '''"""

content = content.replace(old_prompt, new_prompt)

with open("orchestrator/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(content)
