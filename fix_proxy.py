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
            
            # Fallback to HF if Gemini quota is exhausted or failed
            hf_key = os.environ.get("HF_API_KEY", "")
            hf_res = requests.post("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
                                   headers={"Authorization": f"Bearer {hf_key}"},
                                   json={"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": prompt}]})
            if hf_res.status_code == 200:
                return hf_res.json()["choices"][0]["message"]["content"]
            
            # If that fails, try via huggingface_hub client if available
            try:
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=hf_key)
                res = client.chat_completion([{"role": "user", "content": prompt}], model="Qwen/Qwen2.5-72B-Instruct")
                return res.choices[0].message.content
            except Exception:
                pass

            return f"Error: {resp.text}"
        except Exception as e:
            return f"Error: {str(e)}"
"""

    content = re.sub(r'    def invoke\(self, prompt: str\) -> str:.*?return f"Error: \{str\(e\)\}"\n', new_invoke, content, flags=re.DOTALL)
    
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)
