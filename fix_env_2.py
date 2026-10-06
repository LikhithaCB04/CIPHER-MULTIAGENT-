import os
import glob

files = glob.glob('agents/*/*.py') + ['orchestrator/orchestrator.py']
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    # Replace the bad injection with the correct one
    bad_string = 'load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in f else "../.env"))\n'
    if bad_string in content:
        content = content.replace(bad_string, 'load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env" if "agents" in __file__ else "../.env"))\n')
        with open(f, 'w') as file:
            file.write(content)
        print(f"Fixed {f}")
