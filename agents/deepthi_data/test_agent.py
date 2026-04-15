"""
test_agent.py — Run this to test ALL features of your agent BEFORE pushing to Git.

Usage:
  1. Start your agent: uvicorn deepthi_agent:app --port 8001 --reload
  2. In another terminal: python test_agent.py

All tests use the exact same API contract as the orchestrator.
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def call_agent(task_id, description, context="", task_type="data_analysis", priority="medium"):
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "description": description,
        "context": context,
        "priority": priority
    }
    r = requests.post(f"{BASE_URL}/run", json=payload, timeout=300)
    return r.json()

def print_result(test_name, result):
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Status: {result['status']}")
    print(f"Summary: {result['summary']}")
    print(f"Logs: {result['logs']}")
    print(f"Result preview (first 300 chars):")
    print(result['result'][:300])
    print(f"Next agent: {result['next_agent']}")

def run_all_tests():
    print("Starting Deepthi Agent Tests...")
    print(f"Endpoint: {BASE_URL}")

    # Health check
    r = requests.get(f"{BASE_URL}/health")
    print(f"\nHealth check: {r.json()}")

    # ---- TEST 1: EDA with NO data (auto-generates dummy) ----
    result = call_agent(
        task_id="test_001",
        description="analyse this dataset and give me insights"
    )
    print_result("EDA with auto-generated dummy data", result)

    # ---- TEST 2: EDA with real CSV data ----
    # Tiny CSV inline
    csv_data = """name,age,salary,department,left
Alice,28,55000,Engineering,0
Bob,35,72000,Marketing,1
Carol,42,90000,Engineering,0
Dave,26,48000,Sales,1
Eve,31,61000,HR,0
Frank,45,110000,Engineering,0
Grace,29,52000,Marketing,1"""
    result = call_agent(
        task_id="test_002",
        description="explore this employee dataset and tell me key patterns",
        context=csv_data
    )
    print_result("EDA with user-provided CSV", result)

    # ---- TEST 3: Data Cleaning ----
    csv_dirty = """id,age,salary,city
1,25,50000,Mumbai
2,,72000,Delhi
3,35,,Bangalore
2,35,,Bangalore
4,28,45000,Mumbai
5,999,55000,Chennai
6,30,48000,"""
    result = call_agent(
        task_id="test_003",
        description="clean this data, handle missing values with median and remove outliers",
        context=csv_dirty
    )
    print_result("Data Cleaning", result)

    # ---- TEST 4: ML Classification ----
    result = call_agent(
        task_id="test_004",
        description="train a machine learning model to predict churn"
    )
    print_result("Auto-ML Classification (dummy HR data)", result)

    # ---- TEST 5: Clustering ----
    result = call_agent(
        task_id="test_005",
        description="segment the customers into groups and cluster them"
    )
    print_result("Clustering/Segmentation", result)

    # ---- TEST 6: Statistical Analysis ----
    result = call_agent(
        task_id="test_006",
        description="run a statistical analysis and check correlations and distributions"
    )
    print_result("Statistical Analysis", result)

    # ---- TEST 7: Data Pipeline ----
    result = call_agent(
        task_id="test_007",
        description="build a data pipeline to preprocess and transform this data for ml"
    )
    print_result("Data Pipeline", result)

    # ---- TEST 8: Visualisation ----
    result = call_agent(
        task_id="test_008",
        description="visualise the data and create charts"
    )
    print_result("Visualizations", result)

    # ---- TEST 9: Feature Engineering ----
    result = call_agent(
        task_id="test_009",
        description="do feature selection and PCA analysis"
    )
    print_result("Feature Engineering & PCA", result)

    # ---- TEST 10: Generate Dummy Data (explicit) ----
    result = call_agent(
        task_id="test_010",
        description="generate a sample ecommerce dataset and show me what's in it"
    )
    print_result("Generate Dummy Ecommerce Data", result)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("If all statuses are 'success', your agent is ready to push to Git.")
    print("Next steps:")
    print("  1. Run: ollama create deepthi-agent -f Modelfile")
    print("  2. Change model='mistral' to model='deepthi-agent' in deepthi_agent.py")
    print("  3. Copy this folder to: agents/deepthi_data/ in the shared repo")
    print("  4. git add . && git commit -m 'feat: add Deepthi data science agent v2'")
    print("  5. git push")

if __name__ == "__main__":
    run_all_tests()
