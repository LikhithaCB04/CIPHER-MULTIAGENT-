from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:  # pragma: no cover - fallback for environments without the package installed
    class OllamaLLM:
        def __init__(self, model: str):
            self.model = model

        def invoke(self, prompt: str) -> str:
            return f"[mock-fullstack] {self.model} responded to: {prompt[:120]}"

app = FastAPI()
llm = OllamaLLM(model="llama3")


class TaskInput(BaseModel):
    task_id: str
    task_type: str = "fullstack"
    description: str
    context: str = ""
    priority: Optional[str] = "medium"


class TaskOutput(BaseModel):
    task_id: str
    status: str
    result: str
    summary: str
    next_agent: Optional[str] = None
    logs: List[str]


GLASSMORPHISM_HERO = """
export default function Hero() {
  return (
    <section className=\"min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white\">
      <div className=\"mx-auto flex max-w-6xl flex-col items-center justify-center px-6 py-24 text-center\">
        <div className=\"rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm\">Welcome</div>
        <h1 className=\"mt-6 text-5xl font-semibold tracking-tight sm:text-6xl\">Craft polished interfaces with motion.</h1>
        <p className=\"mt-6 max-w-2xl text-lg text-slate-300\">A refined landing experience with soft glass panels and animated reveal.</p>
      </div>
    </section>
  );
}
"""

PRICING_TABLE = """
export default function Pricing() {
  return (
    <div className=\"grid gap-6 p-8 md:grid-cols-3\">
      {['Starter', 'Pro', 'Scale'].map((tier, index) => (
        <div key={tier} className=\"rounded-3xl border border-white/10 bg-slate-900/80 p-8\">
          <h3 className=\"text-xl font-semibold text-white\">{tier}</h3>
          <p className=\"mt-2 text-sm text-slate-400\">Flexible plan for {tier.toLowerCase()} teams.</p>
          <div className=\"mt-6 text-4xl font-semibold text-white\">$19</div>
        </div>
      ))}
    </div>
  );
}
"""

DASHBOARD_SHELL = """
export default function DashboardShell() {
  return (
    <div className=\"flex min-h-screen bg-slate-950 text-white\">
      <aside className=\"w-64 border-r border-white/10 bg-slate-900/80 p-6\">Sidebar</aside>
      <main className=\"flex-1 p-8\">Main content area</main>
    </div>
  );
}
"""

AUTH_FORM = """
export default function AuthForm() {
  return (
    <form className=\"mx-auto max-w-md rounded-3xl border border-white/10 bg-slate-900/80 p-8\">
      <h2 className=\"text-2xl font-semibold text-white\">Sign in</h2>
      <input className=\"mt-4 w-full rounded-xl border border-white/10 bg-slate-800 px-4 py-3 text-white\" placeholder=\"Email\" />
      <input className=\"mt-3 w-full rounded-xl border border-white/10 bg-slate-800 px-4 py-3 text-white\" placeholder=\"Password\" />
      <button className=\"mt-6 w-full rounded-xl bg-cyan-500 px-4 py-3 font-semibold text-slate-950\">Continue</button>
    </form>
  );
}
"""

CARD_GRID_GALLERY = """
export default function Gallery() {
  return (
    <div className=\"grid gap-6 p-8 md:grid-cols-3\">
      {[1, 2, 3].map((item) => (
        <div key={item} className=\"rounded-3xl border border-white/10 bg-slate-900/80 p-6 text-white\">Card {item}</div>
      ))}
    </div>
  );
}
"""

DEV_TOOL_LAYOUT = """
export default function DevLayout() {
  return (
    <div className=\"min-h-screen bg-slate-950 p-6 text-white\">
      <div className=\"rounded-3xl border border-white/10 bg-slate-900/80 p-6\">
        <div className=\"flex items-center justify-between\">
          <h2 className=\"text-xl font-semibold\">Developer Console</h2>
          <div className=\"rounded-full bg-emerald-500/20 px-3 py-1 text-sm text-emerald-300\">Live</div>
        </div>
      </div>
    </div>
  );
}
"""

TEMPLATES = {
    "landing": GLASSMORPHISM_HERO,
    "pricing": PRICING_TABLE,
    "dashboard": DASHBOARD_SHELL,
    "auth": AUTH_FORM,
    "gallery": CARD_GRID_GALLERY,
    "developer": DEV_TOOL_LAYOUT,
}


def select_template(description: str) -> str:
    lowered = description.lower()
    if any(word in lowered for word in ["pricing", "subscription", "plans", "tiers"]):
        return "pricing"
    if any(word in lowered for word in ["dashboard", "analytics", "admin", "panel", "overview"]):
        return "dashboard"
    if any(word in lowered for word in ["login", "signin", "auth", "signup", "register"]):
        return "auth"
    if any(word in lowered for word in ["gallery", "cards", "portfolio", "showcase"]):
        return "gallery"
    if any(word in lowered for word in ["developer", "terminal", "console", "tool", "api"]):
        return "developer"
    return "landing"


@app.post("/run", response_model=TaskOutput)
async def process_task(data: TaskInput):
    template_name = select_template(data.description)
    template = TEMPLATES.get(template_name, GLASSMORPHISM_HERO)
    adapted = template.replace("Craft polished interfaces with motion.", f"{data.description} with elegant motion and responsive polish.")
    adapted = adapted.replace("Main content area", f"{data.description} workspace")
    adapted = adapted.replace("Card 1", f"{data.description} card")
    adapted = adapted.replace("Sign in", f"{data.description.title()}")

    summary = f"Used the {template_name} template and adapted it to the requested UI work."
    logs = [f"Selected template: {template_name}", "Adapted preset layout to task description"]
    return TaskOutput(
        task_id=data.task_id,
        status="success",
        result=adapted,
        summary=summary,
        next_agent=None,
        logs=logs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)
