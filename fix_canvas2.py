import re

with open('frontend/src/CanvasApp.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update initialNodes spacing and layout
old_nodes = """const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 80, y: 50 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 80, y: 250 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 80, y: 450 },  data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 80, y: 650 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 80, y: 850 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];"""
new_nodes = """const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 80, y: 50 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 80, y: 190 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 80, y: 330 }, data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 80, y: 470 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 80, y: 610 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];"""
content = content.replace(old_nodes, new_nodes)

# 2. Update Node dimensions
old_dim = 'w-[280px] h-[160px]'
new_dim = 'w-[260px] h-[120px]'
content = content.replace(old_dim, new_dim)

# 3. Remove ReactFlow Controls
content = re.sub(r'<Controls.*?/>', '', content)

with open('frontend/src/CanvasApp.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
