with open("docker-compose.yml", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "- 5174:5174" in line:
        lines[i] = "    - \"5174:5174\"\n"
    if "8002:8002" in line and not "http" in line:
        lines[i] = "    - \"8002:8002\"\n"

with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.writelines(lines)
