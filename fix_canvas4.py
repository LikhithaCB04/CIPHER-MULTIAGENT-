with open("frontend/src/CanvasApp.tsx", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("frontend/src/CanvasApp.tsx", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip() == "import { getBezierPath } from 'reactflow';":
            continue
        f.write(line)
