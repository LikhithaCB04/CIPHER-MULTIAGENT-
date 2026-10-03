import re

with open('frontend/src/CanvasApp.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Change initialNodes layout
nodes_repl = """const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 80, y: 50 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 80, y: 250 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 80, y: 450 },  data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 80, y: 650 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 80, y: 850 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];"""

content = re.sub(r'const initialNodes: Node\[\] = \[.*?\];', nodes_repl, content, flags=re.DOTALL)

# Remove Top Chrome and Floating Input Pill
content = re.sub(r'\{\/\* Top Chrome \*\/}.*?<\/header>', '', content, flags=re.DOTALL)
content = re.sub(r'\{\/\* Floating Input Pill \*\/}.*?<\/div>\s*<\/div>', '</div>', content, flags=re.DOTALL)

with open('frontend/src/CanvasApp.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
