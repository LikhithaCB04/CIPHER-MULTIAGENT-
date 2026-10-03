import re

with open('agents/deepthi_data/deepthi_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update TaskOutput
content = content.replace('logs: list', 'logs: list\n    files: Optional[list] = None')

# 2. Update run_task file loading logic
old_load = """        if task.context and task.context.strip():
            # Try to parse as CSV string
            try:
                df = pd.read_csv(StringIO(task.context))
                data_source = "user_provided_csv"
                logs.append(f"Loaded user CSV: {df.shape}")
            except Exception:
                pass

            # Try Excel workbook content if the context looks like an uploaded spreadsheet path.
            if df is None:
                try:
                    if os.path.exists(task.context):
                        df = pd.read_excel(task.context, engine="openpyxl")
                        data_source = "user_provided_excel"
                        logs.append(f"Loaded user Excel: {df.shape}")
                except Exception:
                    pass
 
            # Try JSON
            if df is None:
                try:
                    data = json.loads(task.context)
                    df = pd.DataFrame(data)
                    data_source = "user_provided_json"
                    logs.append(f"Loaded user JSON: {df.shape}")
                except Exception:
                    pass"""

new_load = """        if task.context and task.context.strip():
            import base64, io
            try:
                ctx_data = json.loads(task.context)
                if "files" in ctx_data and len(ctx_data["files"]) > 0:
                    file_obj = ctx_data["files"][0]
                    file_name = file_obj.get("name", "")
                    file_data = file_obj.get("data", "")
                    
                    if "base64," in file_data:
                        b64_str = file_data.split("base64,")[1]
                        raw_bytes = base64.b64decode(b64_str)
                        if file_name.endswith(".csv"):
                            df = pd.read_csv(io.BytesIO(raw_bytes))
                            data_source = "user_uploaded_csv"
                        else:
                            df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
                            data_source = "user_uploaded_excel"
                    else:
                        # plain text csv
                        df = pd.read_csv(io.StringIO(file_data))
                        data_source = "user_uploaded_csv"
                        
                    logs.append(f"Loaded {data_source}: {df.shape}")
            except Exception as e:
                pass
                
            if df is None:
                # Fallback to direct parse
                try:
                    df = pd.read_csv(StringIO(task.context))
                    data_source = "user_provided_csv"
                except Exception:
                    pass"""
content = content.replace(old_load, new_load)

# 3. Add files to TaskOutput instantiation
old_taskoutput = """        return TaskOutput(
            task_id=task.task_id,
            status="success",
            result=result,
            summary=summary,
            next_agent=next_agent,
            logs=logs
        )"""
new_taskoutput = """        out_files = []
        if intent in ["clean", "pipeline"] and df is not None:
            import base64, io
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            b64 = base64.b64encode(output.getvalue()).decode('utf-8')
            out_files.append({"name": f"processed_{task.task_id}.xlsx", "data": f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"})
        elif intent == "visualize":
            import glob, base64
            for img_file in glob.glob(f"/tmp/{task.task_id}/*.png"):
                with open(img_file, "rb") as im_f:
                    b64 = base64.b64encode(im_f.read()).decode('utf-8')
                    out_files.append({"name": os.path.basename(img_file), "data": f"data:image/png;base64,{b64}"})

        return TaskOutput(
            task_id=task.task_id,
            status="success",
            result=result,
            summary=summary,
            next_agent=next_agent,
            logs=logs,
            files=out_files if out_files else None
        )"""
content = content.replace(old_taskoutput, new_taskoutput)

with open('agents/deepthi_data/deepthi_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)
