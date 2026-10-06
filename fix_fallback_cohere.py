import re

files = [
    "agents/ai_specialist/agent.py",
    "agents/ayeesha_fullstack/ayeesha_agent.py",
    "orchestrator/orchestrator.py"
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    new_invoke = """    def invoke(self, prompt: str) -> str:
        import requests, os
        payload = {"contents": [{"parts":[{"text": prompt}]}]}
        try:
            resp = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            if resp.status_code != 429:
                try:
                    data = resp.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                except: pass
            
            # Fallback to Cohere if Gemini quota is exhausted
            cohere_key = os.environ.get("COHERE_API_KEY", "")
            if cohere_key:
                co_res = requests.post("https://api.cohere.com/v1/chat",
                                       headers={"Authorization": f"Bearer {cohere_key}", "Content-Type": "application/json"},
                                       json={"model": "command-r-plus", "message": prompt})
                if co_res.status_code == 200:
                    return co_res.json()["text"]

            return f"Error: Gemini quota exhausted and fallback failed. {resp.text}"
        except Exception as e:
            return f"Error: {str(e)}"
"""

    content = re.sub(r'    def invoke\(self, prompt: str\) -> str:.*?return f"Error: \{str\(e\)\}"\n', new_invoke, content, flags=re.DOTALL)
    
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)
