with open("agents/ayeesha_fullstack/ayeesha_agent.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("agents/ayeesha_fullstack/ayeesha_agent.py", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip() == "elif required_files.issubset(files.keys()):":
            continue
        f.write(line)
