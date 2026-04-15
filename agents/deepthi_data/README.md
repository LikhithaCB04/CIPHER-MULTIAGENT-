# Deepthi's Data Science & Analytics Agent

**Port:** 8001 | **Model:** mistral (via Ollama) | **API Contract:** Compatible with `shared/api_contracts/contract.json`

## What This Agent Does

This agent is a fully autonomous data scientist. Give it a task in plain English and it handles everything.

### Features

| Feature | Trigger keywords | What it does |
|---|---|---|
| **EDA** | "analyse", "explore", "insight", "overview" | Shape, types, missing values, correlations, outliers, AI insights |
| **Data Cleaning** | "clean", "fix", "missing", "outlier", "preprocess" | Imputation, deduplication, outlier clipping, type fixing |
| **Auto-ML** | "predict", "classify", "model", "train", "forecast" | Tries 3 models, picks best, explains accuracy in plain English |
| **Clustering** | "cluster", "segment", "group" | Auto-selects optimal k, names clusters with AI |
| **Statistics** | "statistic", "correlation", "distribution", "hypothesis" | Normality tests, Pearson/Spearman, business interpretation |
| **Feature Engineering** | "feature", "pca", "selection", "dimension" | PCA, variance analysis, drop/keep recommendations |
| **Data Pipeline** | "pipeline", "etl", "transform", "workflow" | Clean → Encode → Scale → ML-ready, generates reusable code |
| **Visualization** | "visualize", "chart", "plot", "graph" | Distributions, heatmap, boxplots, categorical bars |
| **Generate Data** | "build", "create", "generate", "dummy" | Auto-generates realistic domain-specific data |

### Dummy Data Domains

When no data is provided, the agent auto-generates realistic data based on keywords:
- `sales/retail/product` → sales dataset (date, product, region, revenue, rating)
- `hr/employee/attrition` → HR dataset (department, salary, performance, churn)
- `ecommerce/order/customer` → ecommerce dataset (orders, values, returns, reviews)
- `health/patient/medical` → health dataset (age, BMI, diagnostics, disease flags)
- anything else → generic numeric/categorical dataset

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running and mistral is pulled
ollama pull mistral

# 3. (Optional but recommended) Create custom fine-tuned model
ollama create deepthi-agent -f Modelfile
# Then change model="mistral" to model="deepthi-agent" in deepthi_agent.py

# 4. Start the agent
uvicorn deepthi_agent:app --port 8001 --reload

# 5. Test everything (in a new terminal)
python test_agent.py
```

---

## API Usage

**Endpoint:** `POST http://localhost:8001/run`

**Input (matches contract.json):**
```json
{
  "task_id": "t1",
  "task_type": "data_analysis",
  "description": "clean this dataset and train a model to predict churn",
  "context": "id,age,salary\n1,28,55000\n2,,72000",
  "priority": "high"
}
```

**Output (matches contract.json):**
```json
{
  "task_id": "t1",
  "status": "success",
  "result": "... full analysis report ...",
  "summary": "Cleaned data and trained RandomForest achieving 87% accuracy",
  "next_agent": null,
  "logs": ["Task received", "Loaded user CSV", "Running EDA", "Task complete"]
}
```

---

## Integration Notes

- **Runs independently** on port 8001 — zero conflict with other agents
- **Does not import** from any other agent file
- **next_agent**: returns `"devops"` for pipeline tasks (data ready to deploy), `null` for analysis tasks
- Charts are saved to `/tmp/agent_charts/` — Ayeesha's dashboard can pick them up from there
- All errors are caught and returned as `status: "error"` — will never crash the orchestrator

---

## Git Push Checklist

- [ ] `uvicorn deepthi_agent:app --port 8001 --reload` — agent starts with no errors
- [ ] `python test_agent.py` — all 10 tests pass
- [ ] Confirm you are NOT importing from `orchestrator/`, `ayeesha/`, `mahima/`, or `likitha/`
- [ ] Copy this folder to `agents/deepthi_data/` in the shared repo
- [ ] `git add agents/deepthi_data/`
- [ ] `git commit -m "feat: Deepthi data science agent v2 - all features"`
- [ ] `git push`
