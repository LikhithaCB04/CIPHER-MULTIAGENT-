import os

files_to_check = []
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            files_to_check.append(os.path.join(root, f))

for path in files_to_check:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    if 'openai/gpt-oss-20b' in content:
        content = content.replace('openai/gpt-oss-20b', 'openai/gpt-oss-20b')
        modified = True
    
    if 'openai/gpt-oss-120b' in content:
        content = content.replace('openai/gpt-oss-120b', 'openai/gpt-oss-120b')
        modified = True
        
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
