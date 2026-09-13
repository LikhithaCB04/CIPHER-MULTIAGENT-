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
  data_science:  { color: '#3B82F6', label: 'deepthi',       num: '01' }, // Deepthi blue
  fullstack:     { color: '#A855F7', label: 'ayeesha',       num: '02' }, // Ayeesha purple
  security:      { color: '#EF4444', label: 'mahima',        num: '03' }, // Mahima red
  devops:        { color: '#22C55E', label: 'likitha',       num: '04' }, // Likitha green
  ai_specialist: { color: '#F59E0B', label: 'ai_specialist', num: '05' }, // AI Specialist amber
};

// ─── CUSTOM NODES ────────────────────────────────────────────────
const AgentNode = ({ data, id }: { data: NodeData; id: string }) => {
  const meta = agentMeta[id] || { color: '#ffffff', label: id, num: '00' };
  
  const isRunning = data.status === 'running';
  const isDone = data.status === 'done';

  const glowStyle = isRunning 
    ? { boxShadow: `0 0 20px ${meta.color}40`, borderColor: meta.color } 
    : isDone 
      ? { borderColor: `${meta.color}80` }
      : { borderColor: 'rgba(255,255,255,0.08)' };

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-transparent !border-none" />
      <div 
        className="relative overflow-hidden transition-all duration-300 w-[260px] h-[160px] flex flex-col"
        style={{
          background: 'rgba(20, 20, 28, 0.85)',
          backdropFilter: 'blur(12px)',
          borderWidth: 1,
          borderStyle: 'solid',
          borderRadius: '16px',
          ...glowStyle
        }}
      >
        {/* Number Watermark */}
        <div className="absolute top-2 right-3 text-6xl font-thin tracking-tighter text-white/5 select-none">
          {meta.num}
        </div>

        {/* Header */}
        <div className="px-4 py-3 flex items-center gap-2 border-b border-white/5 relative z-10">
          <div className="relative flex h-2 w-2 items-center justify-center">
            {isRunning && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ backgroundColor: meta.color }} />
            )}
            <span className="relative inline-flex h-2 w-2 rounded-full transition-colors duration-300" style={{ backgroundColor: isRunning ? meta.color : isDone ? meta.color : '#333' }} />
          </div>
          <span className="font-mono text-[10px] tracking-widest uppercase text-white/50">
            {meta.label}:[{data.status}]
          </span>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-4 flex flex-col overflow-hidden relative z-10">
          {data.status === 'idle' ? (
            <div className="h-full flex items-center justify-center">
              <span className="font-bold text-lg text-white/10 tracking-widest uppercase">
                AWAITING TASK
              </span>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5 overflow-y-auto font-mono text-xs">
              {data.logs.map((log, i) => (
                <motion.div 
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  key={i}
                  className="text-white/70 leading-relaxed truncate"
                >
                  <span style={{ color: meta.color }} className="mr-2 opacity-70">&gt;</span>
                  {log}
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-transparent !border-none" />
    </>
  );
};

// ─── CUSTOM EDGES ────────────────────────────────────────────────
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

  const sourceColor = agentMeta[data?.sourceAgent]?.color || '#ffffff';
  const targetColor = agentMeta[data?.targetAgent]?.color || '#ffffff';
  const isRunning = data?.status === 'running';

  return (
    <>
      <defs>
        <linearGradient id={`grad-${id}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={sourceColor} />
          <stop offset="100%" stopColor={targetColor} />
        </linearGradient>
      </defs>
      <BaseEdge
        path={edgePath}
        style={{
          stroke: `url(#grad-${id})`,
          strokeWidth: 2,
          strokeDasharray: isRunning ? '8 8' : 'none',
          animation: isRunning ? 'dash 1s linear infinite' : 'none',
          opacity: 0.8
        }}
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
  { id: 'data_science',  type: 'agent', position: { x: 100, y: 50 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 500, y: -50 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 900, y: 50 },  data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 300, y: 300 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 700, y: 300 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
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

      {/* Top Chrome */}
      <header className="absolute top-0 w-full z-50 flex items-center justify-between px-6 py-4 pointer-events-none">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-black font-bold tracking-tighter shadow-[0_0_20px_rgba(255,255,255,0.2)] pointer-events-auto">
            C.
          </div>
          <div className="text-sm font-medium tracking-wide text-zinc-200 pointer-events-auto">CIPHER Multi-Agent</div>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/5 bg-black/40 px-3 py-1.5 backdrop-blur-md pointer-events-auto">
          <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`} />
          <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-400">
            ws:[{isConnected ? 'connected' : 'disconnected'}]
          </span>
        </div>
      </header>

      {/* Main Canvas Area */}
      <div className="flex-1 relative z-10">
        <ReactFlow
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
          <Controls position="bottom-left" className="!bottom-24 !left-6 !bg-[#14141C] !border-white/5 !fill-white/70" />
        </ReactFlow>
      </div>

      {/* Floating Input Pill */}
      <div className="absolute bottom-8 w-full flex justify-center pointer-events-none z-50 px-4">
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="pointer-events-auto flex items-center w-full max-w-2xl rounded-full border border-white/10 bg-[rgba(20,20,28,0.7)] p-2 backdrop-blur-2xl shadow-2xl transition-all focus-within:border-white/30 focus-within:shadow-[0_0_30px_rgba(255,255,255,0.05)]"
        >
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            onKeyDown={(e) => e.key === 'Enter' && runTask()}
            placeholder="Instruct the swarm..." 
            className="flex-1 bg-transparent px-4 text-sm font-medium text-zinc-100 placeholder-zinc-500 outline-none" 
          />
          <button 
            onClick={runTask} 
            disabled={!input.trim()}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-zinc-100 text-black transition-transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:hover:scale-100"
          >
            <Send size={16} className="mr-0.5 mt-0.5" />
          </button>
        </motion.div>
      </div>
    </div>
  );
}
