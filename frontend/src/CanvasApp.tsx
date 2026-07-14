import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Edge,
  type Node,
  type Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import { Activity, Bot, Cpu, Sparkles, Workflow } from 'lucide-react';
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
      style: { width: 260, borderRadius: 16, border: '1px solid rgba(255,255,255,0.08)', background: '#09090b' },
    };
    set({ nodes: [...state.nodes, outputNode] });
  },
}));

const agentMeta = {
  deepthi: { label: 'Deepthi', role: 'Data', color: '#60a5fa', accent: 'from-blue-500/30 to-sky-400/10' },
  ayeesha: { label: 'Ayeesha', role: 'Fullstack', color: '#a78bfa', accent: 'from-violet-500/30 to-fuchsia-400/10' },
  mahima: { label: 'Mahima', role: 'Security', color: '#f87171', accent: 'from-rose-500/30 to-orange-400/10' },
  likitha: { label: 'Likitha', role: 'DevOps', color: '#4ade80', accent: 'from-emerald-500/30 to-lime-400/10' },
  ai_specialist: { label: 'AI Specialist', role: 'Specialist', color: '#fb923c', accent: 'from-amber-500/30 to-orange-400/10' },
};

const initialNodes: Node[] = [
  {
    id: 'task',
    type: 'task',
    position: { x: 280, y: 24 },
    data: { label: 'New Task', description: 'Type a task to start the pipeline.' },
    style: { width: 320, borderRadius: 18, border: '1px solid rgba(255,255,255,0.12)', background: '#09090b' },
  },
  ...Object.keys(agentMeta).map((agent, index) => ({
    id: agent,
    type: 'agent',
    position: { x: 80 + index * 180, y: 220 },
    data: { agent, label: agentMeta[agent as keyof typeof agentMeta].label, role: agentMeta[agent as keyof typeof agentMeta].role, status: 'idle', logs: [] },
    style: { width: 160, borderRadius: 18, border: '1px solid rgba(255,255,255,0.12)', background: '#09090b' },
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
  const meta = agentMeta[data.agent as keyof typeof agentMeta];
  const statusColor = data.status === 'running' ? '#fbbf24' : data.status === 'done' ? '#4ade80' : data.status === 'error' ? '#f87171' : '#71717a';
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-white/10 bg-[#09090b] p-3 shadow-2xl shadow-black/30">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`rounded-full bg-gradient-to-br ${meta.accent} p-2`}>
            {data.agent === 'deepthi' ? <Cpu size={14} /> : data.agent === 'ayeesha' ? <Bot size={14} /> : data.agent === 'mahima' ? <Sparkles size={14} /> : data.agent === 'likitha' ? <Workflow size={14} /> : <Activity size={14} />}
          </div>
          <div>
            <div className="text-sm font-semibold text-white">{meta.label}</div>
            <div className="text-[11px] text-zinc-400">{meta.role}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: statusColor, boxShadow: data.status === 'running' ? '0 0 10px currentColor' : 'none' }} />
          <span className="text-[10px] uppercase tracking-[0.25em] text-zinc-500">{data.status}</span>
        </div>
      </div>
      <button onClick={() => setExpanded((v) => !v)} className="mt-3 text-[11px] uppercase tracking-[0.25em] text-zinc-400">{expanded ? 'Hide logs' : 'Show logs'}</button>
      {expanded && (
        <div className="mt-2 max-h-28 overflow-auto rounded-xl border border-white/10 bg-black/40 p-2 text-[11px] leading-5 text-zinc-300">
          {data.logs?.length ? data.logs.map((log: string, index: number) => <div key={index}>• {log}</div>) : <div>No output yet.</div>}
        </div>
      )}
    </motion.div>
  );
};

const TaskNode = ({ data }: { data: any }) => (
  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-white/10 bg-[#09090b] p-4 shadow-2xl shadow-black/30">
    <div className="text-[11px] uppercase tracking-[0.25em] text-zinc-500">Task</div>
    <div className="mt-2 text-sm font-semibold text-white">{data.description}</div>
  </motion.div>
);

const OutputNode = ({ data }: { data: any }) => (
  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-white/10 bg-[#09090b] p-3 shadow-2xl shadow-black/30">
    <div className="text-[11px] uppercase tracking-[0.25em] text-zinc-500">Output</div>
    <div className="mt-2 text-sm font-semibold text-white">{data.label}</div>
    <div className="mt-2 text-xs leading-5 text-zinc-400">{data.summary}</div>
  </motion.div>
);

const nodeTypes = { task: TaskNode, agent: AgentNode, output: OutputNode };

export default function CanvasApp() {
  const { nodes, edges, setNodes, setEdges, updateNode, addOutput } = useCanvasStore();
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Idle');
  const [taskId, setTaskId] = useState('');

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [setNodes, setEdges]);

  const onNodesChange = useCallback((changes: any) => setNodes(applyNodeChanges(changes, nodes)), [nodes, setNodes]);
  const onEdgesChange = useCallback((changes: any) => setEdges(applyEdgeChanges(changes, edges)), [edges, setEdges]);
  const onConnect = useCallback((connection: Connection) => setEdges(addEdge(connection, edges)), [edges, setEdges]);

  useEffect(() => {
    const ws = new WebSocket(`${wsBase}/ws`);
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as StreamEvent;
      const store = useCanvasStore.getState();
      if (payload.event === 'task_received') {
        setStatus('Task received');
        setTaskId(payload.task_id || '');
        store.updateNode('task', { data: { ...(store.nodes.find((node) => node.id === 'task')?.data || {}), description: payload.description || 'Task received' } });
      }
      if (payload.event === 'agent_started') {
        setStatus(`Running ${payload.agent}`);
        store.updateNode(payload.agent || '', { data: { ...(store.nodes.find((node) => node.id === payload.agent)?.data || {}), status: 'running', logs: [...(store.nodes.find((node) => node.id === payload.agent)?.data.logs || []), `Started ${payload.agent}`] } });
      }
      if (payload.event === 'agent_finished') {
        setStatus(`Finished ${payload.agent}`);
        store.updateNode(payload.agent || '', { data: { ...(store.nodes.find((node) => node.id === payload.agent)?.data || {}), status: 'done', logs: [...(store.nodes.find((node) => node.id === payload.agent)?.data.logs || []), payload.result_summary || 'Completed'] } });
        if (payload.result_summary) {
          store.addOutput(payload.agent || '', payload.result_summary || 'Completed');
        }
        if (payload.next_agent && payload.agent) {
          const newEdge: Edge = { id: `edge-${payload.agent}-${payload.next_agent}`, source: payload.agent, target: payload.next_agent, animated: true, style: { stroke: '#60a5fa' } };
          store.setEdges([...store.edges, newEdge]);
        }
      }
      if (payload.event === 'pipeline_complete') {
        setStatus('Pipeline complete');
      }
    };
    return () => ws.close();
  }, []);

  const runTask = async () => {
    if (!input.trim()) return;
    setStatus('Submitting task');
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
    <div className="h-screen w-full bg-[#050505] text-white">
      <div className="flex h-full flex-col">
        <header className="flex items-center justify-between border-b border-white/10 bg-[#09090b]/90 px-6 py-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.35em] text-zinc-500">CIPHER Spatial Canvas</div>
            <div className="text-xl font-semibold">Multi-agent execution view</div>
          </div>
          <div className="rounded-full border border-white/10 bg-black/30 px-4 py-2 text-sm text-zinc-300">{status}</div>
        </header>
        <div className="flex-1">
          <ReactFlow
            nodes={nodeList}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            className="bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.06),_transparent_55%)]"
          >
            <Background color="#3f3f46" gap={24} size={1} />
            <MiniMap nodeColor={() => '#3b82f6'} pannable className="!bottom-6 !right-6 !bg-black/30 !rounded-2xl" />
            <Controls position="bottom-left" />
          </ReactFlow>
        </div>
        <footer className="border-t border-white/10 bg-[#09090b]/90 p-4">
          <div className="flex items-center gap-3">
            <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Describe the work to route across the agents" className="flex-1 rounded-full border border-white/10 bg-black/30 px-4 py-3 text-sm outline-none" />
            <button onClick={runTask} className="rounded-full bg-white px-5 py-3 text-sm font-semibold text-black">Run</button>
          </div>
        </footer>
      </div>
    </div>
  );
}
