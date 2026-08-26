import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Paperclip, ChevronDown, Cpu, Plus, MessageSquare,
  Server, Shield, Database, Code, Cloud, Activity,
  CheckCircle2, XCircle, Loader2, Wifi, WifiOff, Trash2, X, FileText
} from 'lucide-react';

// ─── Agent Definitions ───────────────────────────────────────────────
const AGENTS = [
  { id: 'data_science',  name: 'Data Science',  desc: 'Python & ML pipelines',       icon: Database,   color: '#22d3ee', glow: 'rgba(34,211,238,0.25)' },
  { id: 'fullstack',     name: 'Fullstack',      desc: 'React, Node, APIs',           icon: Code,       color: '#a78bfa', glow: 'rgba(167,139,250,0.25)' },
  { id: 'security',      name: 'Security',       desc: 'Audits & Vulnerability scan', icon: Shield,     color: '#f87171', glow: 'rgba(248,113,113,0.25)' },
  { id: 'devops',        name: 'DevOps',         desc: 'Docker & Cloud infra',        icon: Cloud,      color: '#4ade80', glow: 'rgba(74,222,128,0.25)' },
  { id: 'ai_specialist', name: 'AI Specialist',  desc: 'LLM fine-tuning & prompts',  icon: Server,     color: '#fb923c', glow: 'rgba(251,146,60,0.25)'  },
];

const MODELS = [
  { id: 'orchestrator', name: 'CIPHER Orchestrator', icon: Cpu },
  ...AGENTS.map(a => ({ id: a.id, name: a.name, icon: a.icon })),
];

const TEMPLATES = [
  { emoji: '🔐', title: 'Auth System',   prompt: 'Build a complete login and registration system with JWT authentication, password hashing, and protected routes.' },
  { emoji: '📝', title: 'Todo App',      prompt: 'Build a full stack todo app with React frontend and Node backend where users can add, edit, delete and filter tasks.' },
  { emoji: '🛒', title: 'E-commerce',   prompt: 'Build an e-commerce product listing page with shopping cart, product search, and checkout flow.' },
  { emoji: '📊', title: 'Dashboard',    prompt: 'Build an admin analytics dashboard with charts for user stats, revenue, and activity metrics.' },
  { emoji: '🔍', title: 'Security Audit', prompt: 'Perform a security audit on a web application: check for XSS, SQL injection, CSRF vulnerabilities and suggest fixes.' },
  { emoji: '🤖', title: 'AI Chatbot',   prompt: 'Build an AI-powered chatbot with streaming responses, conversation history, and context management.' },
];

// ─── Types ───────────────────────────────────────────────────────────
type AgentStatus = 'idle' | 'running' | 'done' | 'error';

interface AgentState {
  id: string;
  status: AgentStatus;
  log: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agents_used?: string[];
  results?: any[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

// ─── Helper ──────────────────────────────────────────────────────────
const API = 'http://localhost:8000';

const getAgentMeta = (id: string) => AGENTS.find(a => a.id === id) || AGENTS[4];

// ─── Component ───────────────────────────────────────────────────────
export default function IDE() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<{name: string, content: string}[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeModelId, setActiveModelId] = useState('orchestrator');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [agentStates, setAgentStates] = useState<AgentState[]>(
    AGENTS.map(a => ({ id: a.id, status: 'idle', log: '' }))
  );

  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ── Load sessions from localStorage ──────────────────────────────
  useEffect(() => {
    const saved = localStorage.getItem('cipher_sessions');
    if (saved) {
      const parsed: ChatSession[] = JSON.parse(saved);
      setSessions(parsed);
      if (parsed.length > 0) setCurrentSessionId(parsed[0].id);
      else createNewSession();
    } else {
      createNewSession();
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0)
      localStorage.setItem('cipher_sessions', JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, currentSessionId]);

  // ── Backend health check ──────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(8000) });
        setBackendOnline(r.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── WebSocket for live agent events ──────────────────────────────
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws`);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        const payload = JSON.parse(e.data);
        if (payload.event === 'agent_started') {
          setAgentStates(prev => prev.map(a =>
            a.id === payload.agent ? { ...a, status: 'running', log: `Running: ${payload.description?.slice(0,60) || '...'}` } : a
          ));
        }
        if (payload.event === 'agent_finished') {
          setAgentStates(prev => prev.map(a =>
            a.id === payload.agent ? { ...a, status: 'done', log: payload.result_summary || 'Completed.' } : a
          ));
        }
        if (payload.event === 'pipeline_complete') {
          // Reset all to idle after 4 seconds
          setTimeout(() => {
            setAgentStates(prev => prev.map(a => ({ ...a, status: 'idle' })));
          }, 4000);
        }
      };

      ws.onclose = () => setTimeout(connect, 3000); // auto-reconnect
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── Session helpers ───────────────────────────────────────────────
  const currentSession = sessions.find(s => s.id === currentSessionId) || sessions[0];

  const createNewSession = useCallback(() => {
    const id = Date.now().toString();
    const s: ChatSession = {
      id,
      title: 'New Session',
      messages: [{ id: id + '_0', role: 'assistant', content: "I'm Cipher Orchestrator. Describe what you want to build or run." }],
      updatedAt: Date.now(),
    };
    setSessions(prev => [s, ...prev]);
    setCurrentSessionId(id);
  }, []);

  // ── File handler ──────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      Array.from(e.target.files).forEach(file => {
        const reader = new FileReader();
        reader.onload = (ev) => {
          if (ev.target?.result) {
            setAttachments(prev => [...prev, { name: file.name, content: ev.target!.result as string }]);
          }
        };
        reader.readAsText(file);
      });
      e.target.value = '';
    }
  };

  const deleteSession = (id: string) => {
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id);
      if (id === currentSessionId && next.length > 0) setCurrentSessionId(next[0].id);
      if (next.length === 0) createNewSession();
      return next;
    });
  };

  const pushMessages = useCallback((sessionId: string, messages: ChatMessage[]) => {
    setSessions(prev => prev.map(s => {
      if (s.id !== sessionId) return s;
      let title = s.title;
      if (title === 'New Session') {
        const first = messages.find(m => m.role === 'user');
        if (first) title = first.content.slice(0, 28) + (first.content.length > 28 ? '…' : '');
      }
      return { ...s, messages, title, updatedAt: Date.now() };
    }));
  }, []);

  // ── Send handler ──────────────────────────────────────────────────
  const handleSend = async (text: string = input) => {
    const sid = currentSessionId;

    const userMsg: ChatMessage = { id: Date.now().toString(), role: 'user', content: text };
    const pending = [...currentSession.messages, userMsg];
    pushMessages(sid, pending);
    setInput('');
    setIsLoading(true);

    // Reset agent states for new run
    setAgentStates(prev => prev.map(a => ({ ...a, status: 'idle', log: '' })));

    try {
      const res = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: text, context: '' }),
      });
      const data = await res.json();

      const reply: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `✅ Task complete. ${data.agents_used?.length || 0} agent(s) executed.`,
        agents_used: data.agents_used,
        results: data.results,
      };
      pushMessages(sid, [...pending, reply]);
    } catch {
      const err: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: '⚠️ Could not reach the Cipher backend at localhost:8000. Make sure the orchestrator is running.',
      };
      pushMessages(sid, [...pending, err]);
    } finally {
      setIsLoading(false);
    }
  };

  const activeModel = MODELS.find(m => m.id === activeModelId) || MODELS[0];

  // ─────────────────────────────────────────────────────────────────
  return (
    <div className="h-screen w-full flex bg-[#050505] text-white overflow-hidden" style={{ fontFamily: "'Inter', sans-serif" }}>

      {/* ── Narrow icon rail ─────────────────────────────────────── */}
      <div className="w-12 flex flex-col items-center py-5 gap-6 border-r border-[#1a1a1a] bg-[#080808]">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xs font-black">C</div>
        <div className="flex flex-col gap-4 mt-2">
          {[MessageSquare, Activity, Cpu].map((Icon, i) => (
            <Icon key={i} className={`w-4 h-4 cursor-pointer transition-colors ${i === 0 ? 'text-white' : 'text-[#444] hover:text-[#888]'}`} />
          ))}
        </div>
      </div>

      {/* ── Session sidebar ───────────────────────────────────────── */}
      <div className="w-56 flex flex-col border-r border-[#1a1a1a] bg-[#070707]">
        <div className="px-4 py-3 flex items-center justify-between border-b border-[#1a1a1a]">
          <span className="text-[10px] font-bold tracking-[0.2em] text-[#555] uppercase">Sessions</span>
          <button onClick={createNewSession} className="text-[#555] hover:text-white transition-colors">
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto py-2 space-y-0.5 px-2">
          <AnimatePresence>
            {sessions.map(s => (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, scale: 0.95 }}
                onClick={() => setCurrentSessionId(s.id)}
                className={`group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all text-xs font-mono ${
                  s.id === currentSessionId
                    ? 'bg-[#1a1a1a] text-white'
                    : 'text-[#555] hover:text-[#aaa] hover:bg-[#0f0f0f]'
                }`}
              >
                <span className="truncate flex-1">{s.title}</span>
                <button onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity ml-1 text-[#555] hover:text-rose-400">
                  <Trash2 className="w-3 h-3" />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Chat panel (now in middle) ───────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col bg-[#070707] border-r border-[#1a1a1a]">

        {/* Chat header + model selector */}
        <div className="h-12 border-b border-[#1a1a1a] flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.8)]" />
            <span className="text-[10px] font-bold tracking-[0.2em] text-[#666] uppercase">Cipher Chat</span>
          </div>

          {/* Model dropdown */}
          <div className="relative">
            <button onClick={() => setShowModelDropdown(v => !v)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#1f1f1f] bg-[#0d0d0d] hover:bg-[#151515] transition-colors text-xs font-mono">
              <activeModel.icon className="w-3.5 h-3.5" />
              <span className="text-[#aaa]">{activeModel.name}</span>
              <ChevronDown className="w-3 h-3 text-[#444]" />
            </button>
            <AnimatePresence>
              {showModelDropdown && (
                <motion.div initial={{ opacity: 0, y: 6, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 6, scale: 0.97 }}
                  className="absolute right-0 top-9 w-64 rounded-xl border border-[#1f1f1f] bg-[#0a0a0a] shadow-2xl overflow-hidden z-50 py-1.5">
                  {MODELS.map(m => (
                    <div key={m.id} onClick={() => { setActiveModelId(m.id); setShowModelDropdown(false); }}
                      className="flex items-center gap-3 px-4 py-2.5 hover:bg-[#151515] cursor-pointer transition-colors">
                      <m.icon className="w-3.5 h-3.5 text-[#666]" />
                      <span className="text-xs font-mono text-[#aaa]">{m.name}</span>
                      {activeModelId === m.id && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-500" />}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          <AnimatePresence initial={false}>
            {currentSession?.messages.map(msg => (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`max-w-[88%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-white text-black rounded-tr-sm font-medium'
                    : msg.role === 'system'
                    ? 'bg-rose-950/40 border border-rose-500/20 text-rose-300 rounded-tl-sm font-mono text-xs'
                    : 'bg-[#111] border border-[#1f1f1f] text-[#ccc] rounded-tl-sm'
                }`}>
                  {msg.content}
                </div>

                {/* Agent result cards */}
                {msg.agents_used && msg.agents_used.length > 0 && (
                  <div className="mt-3 w-full space-y-2 max-w-[96%]">
                    <div className="text-[10px] text-[#444] font-mono uppercase tracking-wider flex items-center gap-2">
                      <span className="flex-1 h-px bg-[#1a1a1a]" /> agents executed <span className="flex-1 h-px bg-[#1a1a1a]" />
                    </div>
                    {msg.results?.map((res, idx) => {
                      const agentId = msg.agents_used![idx] || 'ai_specialist';
                      const meta = getAgentMeta(agentId);
                      const Icon = meta.icon;
                      return (
                        <motion.div key={idx} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.08 }}
                          className="p-3 rounded-xl border text-xs font-mono"
                          style={{ borderColor: meta.color + '30', backgroundColor: meta.glow }}>
                          <div className="flex items-center gap-2 mb-2 font-bold uppercase tracking-wider" style={{ color: meta.color }}>
                            <Icon className="w-3 h-3" />{meta.name}
                          </div>
                          {res.error
                            ? <div className="text-rose-400 break-words">{res.error}</div>
                            : <div className="text-[#999] space-y-3">
                                <div>{res.summary || 'Agent completed successfully.'}</div>
                                {res.result && (
                                  <pre className="bg-[#000] p-3 rounded-lg overflow-x-auto text-[10px] text-emerald-400 border border-[#1a1a1a] whitespace-pre-wrap break-words">
                                    <code>{res.result}</code>
                                  </pre>
                                )}
                              </div>
                          }
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 px-4 py-3 w-fit rounded-2xl rounded-tl-sm bg-[#111] border border-[#1f1f1f]">
              {['bg-blue-500','bg-purple-500','bg-cyan-500'].map((c, i) => (
                <span key={i} className={`w-1.5 h-1.5 rounded-full ${c} animate-bounce`}
                  style={{ animationDelay: `${i * 0.1}s`, boxShadow: `0 0 6px currentColor` }} />
              ))}
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Templates row */}
        {(currentSession?.messages.length ?? 0) <= 2 && (
          <div className="px-4 pb-2 flex gap-2 overflow-x-auto scrollbar-hide">
            {TEMPLATES.map((t, i) => (
              <motion.button key={i} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
                onClick={() => handleSend(t.prompt)}
                className="flex-shrink-0 text-left w-36 p-2.5 rounded-xl border border-[#1a1a1a] bg-[#0d0d0d] hover:bg-[#131313] hover:border-[#333] transition-all group">
                <div className="text-base mb-1">{t.emoji}</div>
                <div className="text-[11px] font-semibold text-[#888] group-hover:text-white transition-colors">{t.title}</div>
              </motion.button>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="p-4 border-t border-[#1a1a1a]">
          {/* Attachments preview */}
          {attachments.length > 0 && (
            <div className="flex gap-2 mb-2 flex-wrap">
              {attachments.map((file, i) => (
                <div key={i} className="flex items-center gap-1.5 px-3 py-1.5 bg-[#1a1a1a] rounded-lg text-xs text-white border border-[#333]">
                  <FileText className="w-3 h-3 text-[#888]" />
                  <span className="max-w-[150px] truncate">{file.name}</span>
                  <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))} className="ml-1 text-[#888] hover:text-white">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
          
          <div className="flex items-end gap-2 border border-[#1f1f1f] rounded-2xl p-2 bg-[#0a0a0a] focus-within:border-[#333] focus-within:shadow-[0_0_20px_rgba(255,255,255,0.03)] transition-all">
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" multiple accept="text/*,application/json,text/markdown,.py,.js,.jsx,.ts,.tsx,.html,.css,.csv" />
            <button onClick={() => fileInputRef.current?.click()} className="p-1.5 text-[#444] hover:text-[#888] transition-colors">
              <Paperclip className="w-4 h-4" />
            </button>
            <textarea value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder="Describe a task for the agents… (Shift+Enter for newline)"
              className="flex-1 bg-transparent outline-none text-sm text-white placeholder-[#333] font-mono resize-none min-h-[36px] max-h-28 py-1.5 px-1"
              rows={1}
            />
            <button onClick={() => handleSend()} disabled={isLoading || !input.trim()}
              className="p-2 bg-white rounded-xl text-black hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-[0_0_12px_rgba(255,255,255,0.4)] transition-all">
              <Send className="w-4 h-4" />
            </button>
          </div>
          {!backendOnline && backendOnline !== null && (
            <p className="text-[10px] text-rose-400/70 font-mono mt-2 text-center">
              Backend offline — start the orchestrator: <code>uvicorn orchestrator:app --port 8000</code>
            </p>
          )}
        </div>
      </div>

      {/* ── Agent Status Panel (now on the right) ─────── */}
      <div className="w-[420px] flex flex-col bg-[#050505]">
        {/* Header */}
        <div className="h-12 border-b border-[#1a1a1a] flex items-center justify-between px-5 bg-[#080808]">
          <span className="text-[10px] font-bold tracking-[0.25em] text-[#555] uppercase">Live Agent Monitor</span>
          {/* Backend status indicator */}
          <div className={`flex items-center gap-2 text-[10px] font-mono px-3 py-1 rounded-full border ${
            backendOnline === null ? 'border-[#333] text-[#555]' :
            backendOnline ? 'border-emerald-500/30 text-emerald-400 bg-emerald-400/5' : 'border-rose-500/30 text-rose-400 bg-rose-400/5'
          }`}>
            {backendOnline === null ? <Loader2 className="w-3 h-3 animate-spin" /> :
             backendOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {backendOnline === null ? 'Checking…' : backendOnline ? 'Backend online' : 'Backend offline'}
          </div>
        </div>

        {/* Agent cards grid */}
        <div className="flex-1 p-5 overflow-auto">
          <div className="grid grid-cols-1 gap-3">
            {AGENTS.map(agent => {
              const state = agentStates.find(s => s.id === agent.id)!;
              const Icon = agent.icon;
              return (
                <motion.div
                  key={agent.id}
                  animate={{ boxShadow: state.status === 'running' ? `0 0 20px ${agent.glow}` : '0 0 0 transparent' }}
                  transition={{ duration: 0.5 }}
                  className="p-4 rounded-xl border border-[#1a1a1a] bg-[#090909] flex items-center gap-4"
                >
                  {/* Icon */}
                  <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: agent.glow, border: `1px solid ${agent.color}30` }}>
                    <Icon className="w-4 h-4" style={{ color: agent.color }} />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-sm font-semibold text-white">{agent.name}</span>
                      <span className="text-[10px] text-[#444] font-mono">{agent.desc}</span>
                    </div>
                    <div className="text-[11px] text-[#555] font-mono truncate">
                      {state.log || (state.status === 'idle' ? 'Waiting for task…' : '')}
                    </div>
                  </div>

                  {/* Status badge */}
                  <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full flex-shrink-0 ${
                    state.status === 'running' ? 'bg-yellow-400/10 text-yellow-400' :
                    state.status === 'done'    ? 'bg-emerald-400/10 text-emerald-400' :
                    state.status === 'error'   ? 'bg-rose-400/10 text-rose-400' :
                                                  'bg-[#111] text-[#444]'
                  }`}>
                    {state.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
                    {state.status === 'done'    && <CheckCircle2 className="w-3 h-3" />}
                    {state.status === 'error'   && <XCircle className="w-3 h-3" />}
                    {state.status === 'idle'    && <span className="w-1.5 h-1.5 rounded-full bg-[#333]" />}
                    {state.status}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
