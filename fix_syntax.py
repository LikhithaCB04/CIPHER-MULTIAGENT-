import re

with open('agents/deepthi_data/deepthi_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('logs: list\n    files: Optional[list] = None)', 'logs: list)')

# For TaskOutput:
content = content.replace('    logs: list\n    files: Optional[list] = None\n\n# ===', '    logs: list\n    files: Optional[list] = None\n\n# ===')
content = content.replace('    next_agent: Optional[str] = None\n    logs: list\n', '    next_agent: Optional[str] = None\n    logs: list\n    files: Optional[list] = None\n')


with open('agents/deepthi_data/deepthi_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)
