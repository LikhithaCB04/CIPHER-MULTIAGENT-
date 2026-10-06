with open("docker-compose.yml", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "- 8002:8002" in line:
        if not "- 5174:5174" in lines[i+1]:
            lines.insert(i+1, "      - 5174:5174\n")
        break

with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.writelines(lines)
