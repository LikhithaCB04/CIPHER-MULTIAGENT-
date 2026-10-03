import React, { useState, useEffect, useCallback, useMemo } from 'react';
import type { Connection, Edge, Node } from 'reactflow';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  Handle,
  Position,
  BaseEdge,
  getBezierPath
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Send } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { create } from 'zustand';

// ─── TYPES & STORES ──────────────────────────────────────────────
interface StreamEvent {
  event: string;
  agent?: string;
  task_id?: string;
  description?: string;
  result_summary?: string;
  next_agent?: string;
}

interface NodeData {
  agent: string;
  status: 'idle' | 'running' | 'done';
  logs: string[];
}

interface EdgeData {
  sourceAgent: string;
  targetAgent: string;
  status: 'idle' | 'running' | 'done';
}

interface NodeState {
  nodes: Node[];
  edges: Edge[];
  outputs: Record<string, string[]>;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  updateNode: (id: string, partial: Partial<Node>) => void;
  addOutput: (agent: string, output: string) => void;
}

const useCanvasStore = create<NodeState>((set, get) => ({
  nodes: [],
  edges: [],
  outputs: {},
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  updateNode: (id, partial) =>
    set({
      nodes: get().nodes.map((n) => (n.id === id ? { ...n, ...partial } : n)),
    }),
  addOutput: (agent, output) =>
    set({
      outputs: {
        ...get().outputs,
        [agent]: [...(get().outputs[agent] || []), output],
      },
    }),
}));

// ─── AGENT METADATA ──────────────────────────────────────────────
const agentMeta: Record<string, { color: string; label: string; num: string }> = {
  data_science:  { color: '#00F0FF', label: 'DATA SCIENTIST',   num: '01' }, 
  fullstack:     { color: '#B026FF', label: 'FULLSTACK DEV',    num: '02' }, 
  security:      { color: '#FF003C', label: 'SECURITY AUDIT',   num: '03' }, 
  devops:        { color: '#39FF14', label: 'DEVOPS / CI-CD',   num: '04' }, 
  ai_specialist: { color: '#FFB800', label: 'AI SPECIALIST',    num: '05' }, 
};

// ─── CUSTOM NODES ────────────────────────────────────────────────
const AgentNode = ({ data, id }: { data: NodeData; id: string }) => {
  const meta = agentMeta[id] || { color: '#ffffff', label: id, num: '00' };
  
  const isRunning = data.status === 'running';
  const isDone = data.status === 'done';

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-transparent !border-none" />
      <div 
        className="relative overflow-hidden w-[340px] h-[100px] flex flex-col font-mono transition-all duration-300 transform hover:scale-[1.02]"
        style={{
          background: isRunning ? `linear-gradient(90deg, ${meta.color}20, rgba(10,10,15,0.9))` : 'rgba(10, 10, 15, 0.95)',
          border: `1px solid ${isRunning ? meta.color : 'rgba(255,255,255,0.1)'}`,
          borderLeft: `4px solid ${meta.color}`,
          boxShadow: isRunning ? `0 0 20px ${meta.color}40, inset 0 0 30px ${meta.color}15` : '0 4px 20px rgba(0,0,0,0.5)',
          clipPath: 'polygon(0 0, 100% 0, 100% 80%, 95% 100%, 0 100%)'
        }}
      >
        {/* Cyberpunk Grid / Scanlines */}
        <div 
          className="absolute inset-0 opacity-10 pointer-events-none" 
          style={{ 
            backgroundImage: 'linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, .3) 25%, rgba(255, 255, 255, .3) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .3) 75%, rgba(255, 255, 255, .3) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, .3) 25%, rgba(255, 255, 255, .3) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, .3) 75%, rgba(255, 255, 255, .3) 76%, transparent 77%, transparent)', 
            backgroundSize: '20px 20px' 
          }}
        />
        
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-white/10 bg-black/40 relative z-10">
          <div className="flex items-center gap-3">
             {isRunning ? (
               <div className="relative flex h-2 w-2">
                 <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ backgroundColor: meta.color }}></span>
                 <span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: meta.color }}></span>
               </div>
             ) : (
               <div className="w-2 h-2 rounded-full shadow-[0_0_8px_rgba(255,255,255,0.2)]" style={{ background: isDone ? meta.color : '#333' }} />
             )}
             <span className="text-[12px] font-bold tracking-[0.2em] uppercase" style={{ color: isRunning || isDone ? meta.color : '#888', textShadow: isRunning ? `0 0 10px ${meta.color}` : 'none' }}>
               SYS.{meta.label}
             </span>
          </div>
          <span className="text-[10px] text-white/30 font-bold tracking-widest">[{meta.num}]</span>
        </div>
        
        {/* Body */}
        <div className="flex-1 px-4 flex flex-col justify-center relative z-10">
          {data.status === 'idle' && (
            <div className="text-[11px] text-white/20 uppercase tracking-[0.2em] animate-pulse">
              &gt; SYSTEM_STANDBY
            </div>
          )}
          {data.status === 'running' && (
            <div className="text-[11px] text-white/90 uppercase tracking-wider flex items-center gap-3">
               <svg className="animate-spin h-4 w-4" style={{ color: meta.color }} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
               </svg>
               EXECUTING PROTOCOL...
            </div>
          )}
          {data.status === 'done' && (
            <div className="text-[11px] uppercase tracking-widest font-bold" style={{ color: meta.color }}>
              &gt; TASK_COMPLETED [OK]
            </div>
          )}
        </div>
        
        {/* Bottom decorative bar */}
        <div className="absolute bottom-0 left-0 h-[2px] w-1/3" style={{ background: meta.color, boxShadow: `0 0 10px ${meta.color}` }} />
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-transparent !border-none" />
    </>
  );
};


// --- CUSTOM EDGES ----------------------------------------------------------------
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


// ─── LAYOUT DATA ─────────────────────────────────────────────────
// Scattered "spatial canvas" layout
const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 50, y: 30 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 50, y: 150 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 50, y: 270 }, data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 50, y: 390 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 50, y: 510 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];
const initialEdges: Edge[] = [];

// ─── MAIN APP ────────────────────────────────────────────────────
const apiBase = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const wsBase = apiBase.startsWith('https://')
  ? apiBase.replace('https://', 'wss://')
  : apiBase.startsWith('http://')
    ? apiBase.replace('http://', 'ws://')
    : apiBase;

export default function CanvasApp() {
  const { nodes, edges, setNodes, setEdges } = useCanvasStore();
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Idle');
  const [taskId, setTaskId] = useState('');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [setNodes, setEdges]);

  const onNodesChange = useCallback((changes: any) => setNodes(applyNodeChanges(changes, nodes)), [nodes, setNodes]);
  const onEdgesChange = useCallback((changes: any) => setEdges(applyEdgeChanges(changes, edges)), [edges, setEdges]);
  const onConnect = useCallback((connection: Connection) => setEdges(addEdge({ ...connection, type: 'custom' }, edges)), [edges, setEdges]);

  useEffect(() => {
    let ws: WebSocket;
    const connectWs = () => {
      ws = new WebSocket(`${wsBase}/ws`);
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => setIsConnected(false);
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data) as StreamEvent;
        const store = useCanvasStore.getState();
        
        if (payload.event === 'task_received') {
          setStatus('Processing');
          setTaskId(payload.task_id || '');
        }
        
        if (payload.event === 'agent_started') {
          setStatus(`Routing to ${payload.agent}`);
          store.updateNode(payload.agent || '', { 
            data: { 
              ...(store.nodes.find((node) => node.id === payload.agent)?.data || {}), 
              status: 'running', 
              logs: [...(store.nodes.find((node) => node.id === payload.agent)?.data.logs || []), `Process initiated.`] 
            } 
          });
        }
        
        if (payload.event === 'agent_finished') {
          setStatus(`Completed ${payload.agent}`);
          store.updateNode(payload.agent || '', { 
            data: { 
              ...(store.nodes.find((node) => node.id === payload.agent)?.data || {}), 
              status: 'done', 
              logs: [...(store.nodes.find((node) => node.id === payload.agent)?.data.logs || []), payload.result_summary || 'Task completed successfully.'] 
            } 
          });
          
          if (payload.result_summary) {
            store.addOutput(payload.agent || '', payload.result_summary);
          }
          
          if (payload.next_agent && payload.agent) {
            const newEdge: Edge = { 
              id: `edge-${payload.agent}-${payload.next_agent}-${Date.now()}`, 
              source: payload.agent, 
              target: payload.next_agent, 
              type: 'custom',
              data: { sourceAgent: payload.agent, targetAgent: payload.next_agent, status: 'running' }
            };
            store.setEdges([...store.edges, newEdge]);
          }
        }
        
        if (payload.event === 'pipeline_complete') {
          setStatus('Pipeline execution complete');
          store.setEdges(store.edges.map(e => ({ ...e, data: { ...e.data, status: 'done' } })));
        }
      };
    };
    connectWs();
    return () => ws?.close();
  }, []);

  const runTask = async () => {
    if (!input.trim()) return;
    setStatus('Initializing');
    
    setNodes(nodes.map(n => n.type === 'agent' ? { ...n, data: { ...n.data, status: 'idle', logs: [] } } : n));
    setEdges([]);
    
    const response = await fetch(`${apiBase}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: input, context: '', task_id: taskId || undefined }),
    });
    await response.json();
    setInput('');
  };

  const nodeList = useMemo(() => nodes, [nodes]);

  return (
    <div className="h-screen w-full font-sans flex flex-col relative overflow-hidden bg-[#0A0A0F]">
      
      {/* Animated gradient/wave layer behind dot-grid */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none opacity-[0.06]"
        style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.8), rgba(255,255,255,0) 70%)',
          animation: 'pulse 8s ease-in-out infinite alternate'
        }}
      />
      <style>{`
        @keyframes pulse {
          0% { transform: scale(1) translate(0, 0); opacity: 0.05; }
          100% { transform: scale(1.1) translate(2%, 2%); opacity: 0.08; }
        }
      `}</style>

      

      {/* Main Canvas Area */}
      <div className="flex-1 relative z-10">
        <ReactFlow fitViewOptions={{ padding: 0.1, minZoom: 0.5, maxZoom: 1.5 }} defaultViewport={{ x: 0, y: 0, zoom: 1 }}
          nodes={nodeList}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1A1A24" gap={24} size={1.5} />
          
        </ReactFlow>
      </div>

      </div>
  );
}
