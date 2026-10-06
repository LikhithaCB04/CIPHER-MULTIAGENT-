with open("frontend/src/IDE.tsx", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Update fetch to include history
replacement = """
      const sessionHistory = currentSession?.messages.map(m => ({ role: m.role, content: m.content })).slice(-5) || [];
      const res = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: text, context: contextData, history: sessionHistory }),
      });
"""

content = re.sub(r'      const res = await fetch\(`\$\{API\}/run`, \{\n\s*method: \'POST\',\n\s*headers: \{ \'Content-Type\': \'application/json\' \},\n\s*body: JSON\.stringify\(\{ description: text, context: contextData \}\),\n\s*\}\);', replacement, content, flags=re.DOTALL)

with open("frontend/src/IDE.tsx", "w", encoding="utf-8") as f:
    f.write(content)
