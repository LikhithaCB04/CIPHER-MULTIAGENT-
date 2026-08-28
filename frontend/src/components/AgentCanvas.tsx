import React, { useMemo } from 'react';
import ReactFlow, { Background, Controls, Position, MarkerType, Handle } from 'reactflow';
import type { Edge, Node } from 'reactflow';
import 'reactflow/dist/style.css';
import { Bot, Code, Database, Shield, Cloud, Server, Cpu, Loader2 } from 'lucide-react';

const ICONS: Record<string, any> = {
  orchestrator: Cpu,
  data_science: Database,
  fullstack: Code,
  security: Shield,
  devops: Cloud,
  ai_specialist: Server
};

// Custom Node for Agent
const AgentNode = ({ data }: any) => {
  const Icon = ICONS[data.id] || Bot;
  const isRunning = data.status === 'running';
  
  return (
    <div className={`w-80 rounded-xl overflow-hidden bg-[#0a0a0a]/90 backdrop-blur-md border ${isRunning ? 'border-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.3)]' : 'border-[#222]'} transition-all duration-300`}>
      <Handle type="target" position={Position.Left} className="w-2 h-2 !bg-indigo-500" />
      
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#111] border-b border-[#222]">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-white font-mono">{data.name}</span>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && <Loader2 className="w-3 h-3 text-indigo-400 animate-spin" />}
          <span className={`text-[10px] uppercase font-bold tracking-wider ${isRunning ? 'text-indigo-400' : 'text-[#555]'}`}>
            {data.status}
          </span>
        </div>
      </div>
      
      {/* Terminal Area */}
      <div className="p-3 h-32 overflow-y-auto font-mono text-[10px] text-gray-300 bg-black/50 scrollbar-hide">
        {data.log ? (
          <div className="whitespace-pre-wrap">{data.log}</div>
        ) : (
          <div className="text-[#444] italic">Waiting for instructions...</div>
        )}
      </div>
      
      <Handle type="source" position={Position.Right} className="w-2 h-2 !bg-indigo-500" />
    </div>
  );
};

const nodeTypes = {
  agentNode: AgentNode,
};

interface AgentCanvasProps {
  agentStates: { id: string, status: string, log: string }[];
}

export default function AgentCanvas({ agentStates }: AgentCanvasProps) {
  const nodes: Node[] = useMemo(() => {
    return [
      {
        id: 'orchestrator',
        type: 'agentNode',
        position: { x: 50, y: 300 },
        data: { id: 'orchestrator', name: 'Orchestrator', status: 'idle', log: 'Routing tasks...' }
      },
      ...['fullstack', 'data_science', 'security', 'devops', 'ai_specialist'].map((id, index) => {
        const state = agentStates.find(s => s.id === id) || { status: 'idle', log: '' };
        return {
          id,
          type: 'agentNode',
          position: { x: 500, y: 50 + index * 180 },
          data: {
            id,
            name: id.replace('_', ' ').toUpperCase(),
            status: state.status,
            log: state.log
          }
        };
      })
    ];
  }, [agentStates]);

  const edges: Edge[] = useMemo(() => {
    const runningAgents = agentStates.filter(s => s.status === 'running').map(s => s.id);
    return ['fullstack', 'data_science', 'security', 'devops', 'ai_specialist'].map(id => ({
      id: `e-orch-${id}`,
      source: 'orchestrator',
      target: id,
      animated: runningAgents.includes(id),
      style: { stroke: runningAgents.includes(id) ? '#6366f1' : '#333', strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: runningAgents.includes(id) ? '#6366f1' : '#333',
      },
    }));
  }, [agentStates]);

  return (
    <div className="w-full h-full bg-[#030303]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        defaultEdgeOptions={{ type: 'smoothstep' }}
      >
        <Background color="#1a1a1a" gap={16} />
        <Controls className="!bg-[#111] !border-[#222] !fill-white" />
      </ReactFlow>
    </div>
  );
}
