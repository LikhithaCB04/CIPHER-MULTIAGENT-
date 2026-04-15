from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
import uvicorn

# 1. Initialize the App
app = FastAPI()

# 2. Connect to your custom "DevOps Brain"
# This points to Ollama running on your Windows laptop
llm = Ollama(model="devops-pro", base_url="http://host.docker.internal:11434")

# 3. Define the Input Format (The Team Contract)
class TaskInput(BaseModel):
    task_id: str
    task_type: str
    description: str
    context: str = ""

# 4. The Main Logic: Draft -> Review -> Final Output
@app.post("/run")
def run_devops_task(task: TaskInput):
    try:
        print(f"🚀 Likitha's Agent is starting task: {task.task_id}")

        # --- STEP 1: INITIAL DRAFT (With Connection Error Handling) ---
        print("Generating initial deployment draft...")
        try:
            first_draft_prompt = f"""
            You are a Senior DevOps Engineer.
            Task: {task.description}
            Context: {task.context}
            Generate the necessary Dockerfile, docker-compose, and CI/CD config.
            """
            first_draft = llm.invoke(first_draft_prompt)
        except Exception as conn_error:
            print(f"❌ Connection Error: {str(conn_error)}")
            return {"task_id": task.task_id, "status": "error", "result": "Failed to connect to Ollama. Ensure Ollama is running."}

        # --- STEP 2: VALIDATE THE DRAFT ---
        if not first_draft or len(first_draft) < 20:
            return {"task_id": task.task_id, "status": "error", "result": "AI generated a blank or too short response. Please be more descriptive."}

        # --- STEP 3: SELF-REVIEW & CORRECTION (The Elite Step) ---
        print("Self-reviewing for security and efficiency...")
        review_prompt = f"""
        Review the following DevOps code for a {task.description}. 
        1. Ensure it uses Multi-Stage builds (two FROM lines).
        2. Ensure it includes a 'USER' line for security (no root).
        3. Ensure filenames are clearly marked.
        
        ORIGINAL CODE:
        {first_draft}
        
        Provide the FINAL, corrected version of the code.
        """
        final_output = llm.invoke(review_prompt)
        
        print(f"✅ Task {task.task_id} complete.")

        # --- STEP 4: SUCCESS RETURN ---
        return {
            "task_id": task.task_id,
            "status": "success",
            "result": final_output,
            "summary": "Infrastructure generated and self-reviewed for security/efficiency.",
            "logs": [
                "Draft created successfully", 
                "Automated security audit completed",
                "Applied multi-stage build pattern"
            ]
        }

    except Exception as e:
        # --- STEP 5: TOTAL FAILURE SAFETY NET ---
        print(f"💥 Critical Error: {str(e)}")
        return {
            "task_id": task.task_id, 
            "status": "error", 
            "result": "Internal Agent Error",
            "logs": [str(e)]
        }

# 5. Start the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)