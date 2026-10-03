import re

with open('agents/deepthi_data/deepthi_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix output directory in run_visualization
content = content.replace('output_dir = "/tmp/agent_charts"', 'output_dir = f"/tmp/{os.environ.get(\'TASK_ID_HACK\', \'agent_charts\')}"')

# 2. Add custom_dashboard intent
old_routing = """        intent = detect_task_intent(task.description)
        logs.append(f"Detected task intent: {intent} | Data source: {data_source}")"""
        
new_routing = """        intent = detect_task_intent(task.description)
        desc_lower = task.description.lower()
        if "dashboard" in desc_lower and ("clean" in desc_lower or "sort" in desc_lower or "order" in desc_lower):
            intent = "custom_dashboard"
        logs.append(f"Detected task intent: {intent} | Data source: {data_source}")
        os.environ["TASK_ID_HACK"] = task.task_id"""
content = content.replace(old_routing, new_routing)

# 3. Add custom_dashboard branch
old_eda = """        if intent == "eda":"""
new_eda = """        if intent == "custom_dashboard":
            config = CleaningConfig()
            if "median" in desc_lower: config.strategy_missing = "median"
            elif "mode" in desc_lower: config.strategy_missing = "mode"
            elif "drop" in desc_lower: config.strategy_missing = "drop"
            
            df_clean, cleaning_result = run_cleaning(df, config, logs)
            
            # Basic sort logic
            for col in df_clean.columns:
                if col.lower() in desc_lower:
                    df_clean = df_clean.sort_values(by=col, ascending=("desc" not in desc_lower))
                    logs.append(f"Sorted data by {col}")
                    break
            
            df = df_clean
            visualize_result = run_visualization(df, task.description, logs)
            result = f"Custom Analytics Pipeline Executed.\\n\\n{cleaning_result}\\n\\n{visualize_result}"

        elif intent == "eda":"""
content = content.replace(old_eda, new_eda)

# 4. Also make sure the files return logic works for custom_dashboard
content = content.replace('if intent in ["clean", "pipeline"] and df is not None:', 'if intent in ["clean", "pipeline", "custom_dashboard"] and df is not None:')
content = content.replace('elif intent == "visualize":', 'if intent in ["visualize", "custom_dashboard"]:')

with open('agents/deepthi_data/deepthi_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)
