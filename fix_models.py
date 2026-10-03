import os
import glob

files_to_check = []
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            files_to_check.append(os.path.join(root, f))

for path in files_to_check:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'llama3-70b-8192' in content:
        content = content.replace('llama3-70b-8192', 'llama3-70b-8192')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
