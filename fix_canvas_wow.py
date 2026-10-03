import re

with open('frontend/src/CanvasApp.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Agent Meta
old_meta = """const agentMeta: Record<string, { color: string; label: string; num: string }> = {
  data_science:  { color: '#3B82F6', label: 'deepthi',       num: '01' }, // Deepthi blue
  fullstack:     { color: '#A855F7', label: 'ayeesha',       num: '02' }, // Ayeesha purple
  security:      { color: '#EF4444', label: 'mahima',        num: '03' }, // Mahima red
  devops:        { color: '#22C55E', label: 'likitha',       num: '04' }, // Likitha green
  ai_specialist: { color: '#F59E0B', label: 'ai_specialist', num: '05' }, // AI Specialist amber
};"""
new_meta = """const agentMeta: Record<string, { color: string; label: string; num: string }> = {
  data_science:  { color: '#00F0FF', label: 'DATA SCIENTIST',   num: '01' }, 
  fullstack:     { color: '#B026FF', label: 'FULLSTACK DEV',    num: '02' }, 
  security:      { color: '#FF003C', label: 'SECURITY AUDIT',   num: '03' }, 
  devops:        { color: '#39FF14', label: 'DEVOPS / CI-CD',   num: '04' }, 
  ai_specialist: { color: '#FFB800', label: 'AI SPECIALIST',    num: '05' }, 
};"""
content = content.replace(old_meta, new_meta)

# 2. Update initialNodes
old_nodes = """const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 80, y: 50 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 80, y: 190 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 80, y: 330 }, data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 80, y: 470 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 80, y: 610 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];"""
new_nodes = """const initialNodes: Node[] = [
  { id: 'data_science',  type: 'agent', position: { x: 50, y: 30 },  data: { agent: 'data_science', status: 'idle', logs: [] } },
  { id: 'fullstack',     type: 'agent', position: { x: 50, y: 150 }, data: { agent: 'fullstack', status: 'idle', logs: [] } },
  { id: 'security',      type: 'agent', position: { x: 50, y: 270 }, data: { agent: 'security', status: 'idle', logs: [] } },
  { id: 'devops',        type: 'agent', position: { x: 50, y: 390 }, data: { agent: 'devops', status: 'idle', logs: [] } },
  { id: 'ai_specialist', type: 'agent', position: { x: 50, y: 510 }, data: { agent: 'ai_specialist', status: 'idle', logs: [] } },
];"""
content = content.replace(old_nodes, new_nodes)

# 3. Replace AgentNode component
old_agent_node = re.search(r'const AgentNode = \(\{ data, id \}: \{ data: NodeData; id: string \}\) => \{.*?(?=const edgeTypes)', content, re.DOTALL).group(0)

new_agent_node = """const AgentNode = ({ data, id }: { data: NodeData; id: string }) => {
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
              > SYSTEM_STANDBY
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
              > TASK_COMPLETED [OK]
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

"""
content = content.replace(old_agent_node, new_agent_node)

# 4. Modify fitViewOptions in ReactFlow to prevent extreme zooming and cutting off
old_rf = '<ReactFlow'
new_rf = '<ReactFlow fitViewOptions={{ padding: 0.1, minZoom: 0.5, maxZoom: 1.5 }} defaultViewport={{ x: 0, y: 0, zoom: 1 }}'
content = content.replace(old_rf, new_rf)

with open('frontend/src/CanvasApp.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
