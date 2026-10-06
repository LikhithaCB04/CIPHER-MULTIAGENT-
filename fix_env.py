import os
import glob

files = glob.glob('agents/*/*.py') + ['orchestrator/orchestrator.py']
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    if 'load_dotenv' not in content:
        with open(f, 'w') as file:
            file.write('import os\nfrom dotenv import load_dotenv\nload_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in f else "../.env"))\n\n' + content)
            print(f"Updated {f}")
