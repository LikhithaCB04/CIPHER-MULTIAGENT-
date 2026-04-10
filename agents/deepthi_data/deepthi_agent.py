from langchain_community.llms import Ollama
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from io import StringIO

app = FastAPI()

# We use phi3 because it is lightweight and won't crash your laptop
llm = Ollama(model="tinyllama")

class Task(BaseModel):
    task_id: str
    description: str
    context: str = ""  # This is where the CSV data or file path goes

@app.post('/run')
def run_analysis(task: Task):
    print(f"Deepthi's Agent received task: {task.description}")
    
    # STEP 1: Ask the AI what kind of analysis is needed
    plan_prompt = f"Given this data task: {task.description}, list 3 short steps to analyze the data. Return only a brief list."
    plan = llm.invoke(plan_prompt)

    # STEP 2: Load and Analyze the data
    summary = ""
    if task.context:
        try:
            # We assume the context is a CSV string. We use StringIO to read it like a file.
            df = pd.read_csv(StringIO(task.context))
            # Get basic stats: mean, max, min, etc.
            summary = df.describe().to_string()
        except Exception as e:
            summary = f"Error reading data: {str(e)}"
    else:
        summary = "No data provided in context."

    # STEP 3: Generate a final report using the AI
    report_prompt = f"Data Summary:\n{summary}\n\nTask: {task.description}\n\nWrite a 2-paragraph professional analysis report based on this data."
    final_report = llm.invoke(report_prompt)

    # STEP 4: Return the result in the EXACT format of contract.json
    return {
        "task_id": task.task_id,
        "status": "success",
        "result": final_report,
        "summary": plan,
        "next_agent": "fullstack", # After analysis, we usually send it to Ayeesha to build a dashboard
        "logs": ["Data loaded", "Stats calculated", "Report generated"]
    }
