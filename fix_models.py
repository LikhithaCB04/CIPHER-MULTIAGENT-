import glob
for f in glob.glob('agents/**/*.py', recursive=True) + ['orchestrator/orchestrator.py']:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    if 'gemini-3.5-flash' in content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content.replace('gemini-3.5-flash', 'gemini-3.8-flash'))
        print(f'Fixed {f}')
