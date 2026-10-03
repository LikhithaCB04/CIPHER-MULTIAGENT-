with open("frontend/src/CanvasApp.tsx", "r", encoding="utf-8") as f:
    content = f.read()

restore_code = """
// --- CUSTOM EDGES ----------------------------------------------------------------
import { getBezierPath } from 'reactflow';
const AnimatedEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: any) => {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const isRunning = data?.status === 'running';
  const isDone = data?.status === 'done';

  return (
    <>
      <path
        id={id}
        style={{
          stroke: isRunning ? '#3B82F6' : isDone ? '#444' : '#222',
          strokeWidth: isRunning ? 2 : 1,
          animation: isRunning ? 'dash 1s linear infinite' : 'none',
          strokeDasharray: isRunning ? '8, 8' : 'none',
          transition: 'stroke 0.3s',
        }}
        className="react-flow__edge-path"
        d={edgePath}
      />
      {isRunning && (
        <style>
          {`
            @keyframes dash {
              from { stroke-dashoffset: 16; }
              to { stroke-dashoffset: 0; }
            }
          `}
        </style>
      )}
    </>
  );
};

const nodeTypes = { agent: AgentNode };
const edgeTypes = { custom: AnimatedEdge };
"""

# Replace the single existing edgeTypes line with the restored code
content = content.replace("const edgeTypes = { custom: AnimatedEdge };", restore_code)

with open("frontend/src/CanvasApp.tsx", "w", encoding="utf-8") as f:
    f.write(content)
