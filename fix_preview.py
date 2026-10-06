with open("agents/ayeesha_fullstack/ayeesha_agent.py", "r", encoding="utf-8") as f:
    content = f.read()

# Change 127.0.0.1 to 0.0.0.0 for Vite bind
content = content.replace('"127.0.0.1",\n                                        "--port",', '"0.0.0.0",\n                                        "--port",')
# Change the healthcheck internal fetch to use localhost since it's testing from inside the container
content = content.replace("urllib.request.urlopen(\"http://127.0.0.1:5174\")", "urllib.request.urlopen(\"http://localhost:5174\")")

with open("agents/ayeesha_fullstack/ayeesha_agent.py", "w", encoding="utf-8") as f:
    f.write(content)

with open("docker-compose.yml", "r", encoding="utf-8") as f:
    dc = f.read()

import re
# Add 5174:5174 to ayesha-agent ports
new_ports = """      ports:
      - 8002:8002
      - 5174:5174"""
dc = re.sub(r'      ports:\n      - 8002:8002', new_ports, dc)

with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.write(dc)

