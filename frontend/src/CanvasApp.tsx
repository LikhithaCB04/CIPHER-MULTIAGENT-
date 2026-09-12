import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  BaseEdge,
  getBezierPath,
  type Edge,
  type Node,
  type Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import { Database, Code2, Shield, Server, Sparkles, Send, CheckCircle2, AlertCircle, Circle, Loader2 } from 'lucide-react';
import { create } from 'zustand';

interface StreamEvent {
  event: string;
  task_id?: string;
  description?: string;
  agent?: string;
  result_summary?: string;
  next_agent?: string | null;
}

interface NodeState {
  nodes: Node[];
  edges: Edge[];
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  updateNode: (id: string, changes: Partial<Node>) => void;
  addOutput: (agent: string, summary: string) => void;
}

const useCanvasStore = create<NodeState>((set, get) => ({
  nodes: [],
  edges: [],
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  updateNode: (id, changes) => set((state) => ({
    nodes: state.nodes.map((node) => node.id === id ? { ...node, ...changes } : node),
  })),
  addOutput: (agent, summary) => {
    const state = get();
    const outputId = `${agent}-output-${Date.now()}`;
    const outputNode: Node = {
      id: outputId,
      type: 'output',
      position: { x: 120 + state.nodes.filter((n) => n.type === 'output').length * 140, y: 420 },
      data: { label: `${agent} output`, summary },
      style: { width: 300, background: 'transparent', border: 'none' },
    };
    set({ nodes: [...state.nodes, outputNode] });
  },
}));

const agentMeta: Record<string, any> = {
  deepthi: { label: 'Deepthi', role: 'Data Analysis', color: 'var(--accent-deepthi)', Icon: Database },
  ayeesha: { label: 'Ayeesha', role: 'Fullstack Dev', color: 'var(--accent-ayeesha)', Icon: Code2 },
  mahima: { label: 'Mahima', role: 'Security Audit', color: 'var(--accent-mahima)', Icon: Shield },
  likitha: { label: 'Likitha', role: 'DevOps & Infra', color: 'var(--accent-likitha)', Icon: Server },
  ai_specialist: { label: 'AI Specialist', role: 'LLM Ops', color: 'var(--accent-ai-specialist)', Icon: Sparkles },
};

const initialNodes: Node[] = [
  ...Object.keys(agentMeta).map((agent, index) => ({
    id: agent,
    type: 'agent',
    position: { x: 80 + index * 260, y: 150 },
    data: { agent, status: 'idle', logs: [] },
    style: { width: 220, background: 'transparent', border: 'none' },
  })) as Node[],
];

const initialEdges: Edge[] = [];
const apiBase = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const wsBase = apiBase.startsWith('https://')
  ? apiBase.replace('https://', 'wss://')
  : apiBase.startsWith('http://')
    ? apiBase.replace('http://', 'ws://')
    : apiBase;

const AgentNode = ({ data }: { data: any }) => {
  const [expanded, setExpanded] = useState(true);
  const meta = agentMeta[data.agent] || { label: 'Unknown', role: 'Agent', color: '#fff', Icon: Circle };
  const Icon = meta.Icon;
  
  const isRunning = data.status === 'running';
  const isDone = data.status === 'done';
  const isError = data.status === 'error';
  const isIdle = data.status === 'idle';

  let boxStyle = {};
  let opacity = 1;
  let statusIcon = null;

  if (isIdle) {
    opacity = 0.7;
    boxStyle = { borderColor: meta.color, borderWidth: '1px' };
  } else if (isRunning) {
    boxStyle = { 
      borderColor: meta.color, 
      boxShadow: `0 0 24px -4px ${meta.color}60, inset 0 0 12px -4px ${meta.color}30` 
    };
    statusIcon = <Loader2 size={14} className="animate-spin" style={{ color: meta.color }} />;
  } else if (isDone) {
    boxStyle = { borderTop: `2px solid ${meta.color}` };
    statusIcon = <CheckCircle2 size={14} style={{ color: meta.color }} />;
  } else if (isError) {
    boxStyle = { 
      borderColor: '#ef4444', 
      boxShadow: '0 0 24px -4px rgba(239, 68, 68, 0.4)' 
    };
    statusIcon = <AlertCircle size={14} className="text-red-500" />;
  }

  return (
    <motion.div 
      initial={false}
      animate={{ opacity }}
      transition={{ duration: 0.3 }}
      style={boxStyle}
      className="canvas-card p-4 transition-shadow duration-300 relative overflow-hidden font-ui"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-black/40 border border-white/5">
            <Icon size={16} style={{ color: meta.color }} />
          </div>
          <div>
            <div className="text-sm font-medium tracking-wide text-zinc-100">{meta.label}</div>
            <div className="text-[11px] font-medium text-zinc-500">{meta.role}</div>
          </div>
        </div>
        <div className="flex h-6 w-6 items-center justify-center">
          {statusIcon || <div className="h-2 w-2 rounded-full" style={{ backgroundColor: meta.color }} />}
        </div>
      </div>
      
      <button 
        onClick={() => setExpanded((v) => !v)} 
        className="text-[10px] uppercase tracking-widest text-zinc-500 hover:text-zinc-300 transition-colors mb-2"
      >
        {expanded ? 'Hide Trace' : 'Show Trace'}
      </button>
      
      {expanded && (
        <div className="max-h-32 overflow-y-auto rounded-lg border border-white/5 bg-black/50 p-3 font-code text-[11px] leading-relaxed text-zinc-400">
          {data.logs?.length ? data.logs.map((log: string, index: number) => (
            <div key={index} className="mb-1 last:mb-0 break-words">
              <span className="text-zinc-600 mr-2">›</span>
              {log}
            </div>
          )) : <div className="text-zinc-600 italic">Awaiting execution...</div>}
        </div>
      )}
    </motion.div>
  );
};

const OutputNode = ({ data }: { data: any }) => (
  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="canvas-card p-4 font-ui">
    <div className="text-[10px] uppercase tracking-widest text-zinc-500 mb-2">Final Output</div>
    <div className="text-sm font-medium text-zinc-200 mb-3">{data.label}</div>
    <div className="rounded-lg bg-black/30 p-3 font-code text-[11px] leading-relaxed text-zinc-400 border border-white/5">
      {data.summary}
    </div>
  </motion.div>
);

const AnimatedEdge = ({
  id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, data
}: any) => {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const sourceColor = agentMeta[data?.sourceAgent]?.color || '#ffffff';
  const targetColor = agentMeta[data?.targetAgent]?.color || '#ffffff';
  const isRunning = data?.status === 'running';

  return (
    <>
      <defs>
        <linearGradient id={`grad-${id}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={sourceColor} />
          <stop offset="100%" stopColor={targetColor} />
        </linearGradient>
      </defs>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: `url(#grad-${id})`,
          strokeWidth: 2,
          strokeDasharray: isRunning ? '6 6' : 'none',
          animation: isRunning ? 'dash 1s linear infinite' : 'none',
          opacity: 0.8
        }}
      />
      {isRunning && (
        <style>
          {`
            @keyframes dash {
              from { stroke-dashoffset: 12; }
              to { stroke-dashoffset: 0; }
            }
          `}
        </style>
      )}
    </>
  );
};

const nodeTypes = { agent: AgentNode, output: OutputNode };
const edgeTypes = { custom: AnimatedEdge };

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
          // Update all edges to static when done
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
    
    // Reset nodes to idle
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
    <div className="h-screen w-full font-ui flex flex-col relative" style={{ backgroundColor: 'var(--bg-canvas)' }}>
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
          <span className="text-[11px] font-medium tracking-wide text-zinc-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Main Canvas Area */}
      <div className="flex-1 relative">
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
          <Background color="var(--bg-dot)" gap={20} size={1.5} />
          <MiniMap 
            nodeColor={(n) => {
              if (n.type === 'agent') return agentMeta[n.data.agent]?.color || '#555';
              return '#333';
            }} 
            maskColor="rgba(10, 10, 15, 0.7)"
            className="!bottom-28 !right-6 !bg-[#0A0A0F] !border !border-white/5 !rounded-xl !overflow-hidden" 
          />
          <Controls position="bottom-left" className="!bottom-24 !left-6" />
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
