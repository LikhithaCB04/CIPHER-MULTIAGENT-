
app = FastAPI()

from groq import Groq
import os
import asyncio

class GroqProxy:
    def __init__(self, model_name, max_tokens):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_name
        self.max_tokens = max_tokens
        
    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
        
    async def astream(self, prompt):
        response = await asyncio.to_thread(self.invoke, prompt)
        yield response

llm = GroqProxy(model_name="llama-3.1-70b-versatile", max_tokens=800)
