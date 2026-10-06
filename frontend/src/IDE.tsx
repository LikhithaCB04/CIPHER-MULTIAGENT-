import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Paperclip, ChevronDown, Cpu, Plus, MessageSquare,
  Server, Shield, Database, Code, Cloud, Activity,
  CheckCircle2, XCircle, Loader2, Wifi, WifiOff, Trash2, X, FileText,
  Copy, ThumbsUp, ThumbsDown, Mic, ArrowRight, User
} from 'lucide-react';
import CanvasApp from './CanvasApp';

// ─── Agent Definitions ───────────────────────────────────────────────
const AGENTS = [
  { id: 'data_science',  name: 'Data Science',  desc: 'Python & ML pipelines',       icon: Database,   color: 'var(--accent-deepthi)', glow: 'rgba(176,141,87,0.25)' },
  { id: 'fullstack',     name: 'Fullstack',      desc: 'React, Node, APIs',           icon: Code,       color: 'var(--accent-ayeesha)', glow: 'rgba(107,105,101,0.25)' },
  { id: 'security',      name: 'Security',       desc: 'Audits & Vulnerability scan', icon: Shield,     color: 'var(--accent-mahima)', glow: 'rgba(87,62,46,0.25)' },
  { id: 'devops',        name: 'DevOps',         desc: 'Docker & Cloud infra',        icon: Cloud,      color: 'var(--accent-likitha)', glow: 'rgba(141,123,104,0.25)' },
  { id: 'ai_specialist', name: 'AI Specialist',  desc: 'LLM fine-tuning & prompts',  icon: Server,     color: 'var(--accent-ai-specialist)', glow: 'rgba(201,169,126,0.25)'  },
];

const MODELS = [
  { id: 'orchestrator', name: 'CIPHER Orchestrator', icon: Cpu },
  ...AGENTS.map(a => ({ id: a.id, name: a.name, icon: a.icon })),
];

const TEMPLATES = [
  { emoji: '🔐', title: 'Auth System',   prompt: 'Build a complete login and registration system with JWT authentication, password hashing, and protected routes.', color: 'var(--accent-mahima)', glow: 'rgba(87,62,46,0.1)' },
  { emoji: '📝', title: 'Todo App',      prompt: 'Build a full stack todo app with React frontend and Node backend where users can add, edit, delete and filter tasks.', color: 'var(--accent-ayeesha)', glow: 'rgba(107,105,101,0.1)' },
  { emoji: '🛒', title: 'E-commerce',   prompt: 'Build an e-commerce product listing page with shopping cart, product search, and checkout flow.', color: 'var(--accent-likitha)', glow: 'rgba(141,123,104,0.1)' },
  { emoji: '📊', title: 'Dashboard',    prompt: 'Build an admin analytics dashboard with charts for user stats, revenue, and activity metrics.', color: 'var(--accent-deepthi)', glow: 'rgba(176,141,87,0.1)' },
  { emoji: '🔍', title: 'Security Audit', prompt: 'Perform a security audit on a web application: check for XSS, SQL injection, CSRF vulnerabilities and suggest fixes.', color: 'var(--accent-mahima)', glow: 'rgba(87,62,46,0.1)' },
  { emoji: '🤖', title: 'AI Chatbot',   prompt: 'Build an AI-powered chatbot with streaming responses, conversation history, and context management.', color: 'var(--accent-ai-specialist)', glow: 'rgba(201,169,126,0.1)' },
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
  attachments?: {name: string, content: string}[];
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
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [agentStates, setAgentStates] = useState<AgentState[]>(
    AGENTS.map(a => ({ id: a.id, status: 'idle', log: '' }))
  );

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  // ─── WebSocket for live agent events ──────────────────────────────
  const [activeConfirmations, setActiveConfirmations] = useState<any[]>([]);

  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleRecording = () => {
    if (isRecording && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    
    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        setInput(prev => prev + (prev.endsWith(' ') || prev.length === 0 ? '' : ' ') + finalTranscript);
      }
    };

    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);

    recognition.start();
    recognitionRef.current = recognition;
    setIsRecording(true);
  };

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
        if (payload.event === 'agent_thought') {
          setAgentStates(prev => prev.map(a =>
            a.id === payload.agent ? { ...a, status: 'running', log: payload.text } : a
          ));
        }
        if (payload.event === 'confirmation_required') {
          setActiveConfirmations(prev => [...prev, payload]);
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

  const handleConfirm = (taskId: string, status: 'approved' | 'denied') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'confirmation_response', task_id: taskId, status }));
    }
    setActiveConfirmations(prev => prev.filter(c => c.task_id !== taskId));
  };

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
        if (file.type.startsWith('image/') || file.name.toLowerCase().endsWith('.xlsx') || file.name.toLowerCase().endsWith('.xls')) {
          reader.readAsDataURL(file);
        } else {
          reader.readAsText(file);
        }
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

    const userMsg: ChatMessage = { 
      id: Date.now().toString(), 
      role: 'user', 
      content: text,
      attachments: attachments.length > 0 ? [...attachments] : undefined
    };
    const pending = [...currentSession.messages, userMsg];
    pushMessages(sid, pending);
    setInput('');
    setIsLoading(true);

    // Reset agent states for new run
    setAgentStates(prev => prev.map(a => ({ ...a, status: 'idle', log: '' })));

    try {
      const contextData = attachments.length > 0 ? JSON.stringify({ files: attachments }) : '';
      setAttachments([]);
      

      const sessionHistory = currentSession?.messages.map(m => ({ role: m.role, content: m.content })).slice(-5) || [];
      const res = await fetch(`${API}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: text, context: contextData, history: sessionHistory }),
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
    <div className="h-screen w-full flex bg-surface text-textMain overflow-hidden" style={{ fontFamily: "'Inter', sans-serif" }}>
      
      {activeConfirmations.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-surface border border-surfaceBorder p-6 rounded-xl shadow-2xl max-w-lg w-full">
            <h2 className="text-xl font-bold text-textMain flex items-center gap-2 mb-4">
              <Shield className="w-6 h-6 text-red-500" /> Security Confirmation Required
            </h2>
            <div className="bg-surface p-4 rounded-lg border border-surfaceBorder mb-6">
              <p className="text-sm text-textMuted mb-1">Agent:</p>
              <p className="font-mono text-textMuted mb-4">{activeConfirmations[0].agent}</p>
              <p className="text-sm text-textMuted mb-1">Tool:</p>
              <p className="font-mono text-textMuted mb-4">{activeConfirmations[0].tool}</p>
              <p className="text-sm text-textMuted mb-1">Action:</p>
              <pre className="font-mono text-textMuted text-sm whitespace-pre-wrap">{activeConfirmations[0].action}</pre>
            </div>
            <div className="flex gap-4">
              <button 
                onClick={() => handleConfirm(activeConfirmations[0].task_id, 'denied')}
                className="flex-1 px-4 py-2 bg-transparent border border-red-500 text-red-500 rounded-lg hover:bg-red-500/10 transition-colors"
              >
                Deny
              </button>
              <button 
                onClick={() => handleConfirm(activeConfirmations[0].task_id, 'approved')}
                className="flex-1 px-4 py-2 bg-green-600 text-textMain rounded-lg hover:bg-green-700 transition-colors"
              >
                Approve
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Narrow icon rail ─────────────────────────────────────── */}
      <div className="w-12 flex flex-col items-center py-5 border-r border-surfaceBorder bg-surface">
        <div className="w-7 h-7 rounded-lg bg-surfaceBorder text-white flex items-center justify-center text-xs font-black mb-6">C</div>
        <div className="flex flex-col gap-4">
          {[MessageSquare, Activity, Cpu].map((Icon, i) => (
            <Icon key={i} className={`w-4 h-4 cursor-pointer transition-colors ${i === 0 ? 'text-textMain' : 'text-textMuted hover:text-textMuted'}`} />
          ))}
        </div>
        <div className="mt-auto pt-4">
          <div className="w-8 h-8 rounded-full bg-surfaceBorder border border-surfaceBorder flex items-center justify-center text-textMuted cursor-pointer hover:text-white transition-colors" title="User Profile">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* ── Session sidebar ───────────────────────────────────────── */}
      <div className="w-56 flex flex-col border-r border-surfaceBorder bg-surface">
        <div className="px-4 py-3 flex items-center justify-between border-b border-surfaceBorder">
          <span className="text-[10px] font-bold tracking-[0.2em] text-white uppercase">Projects</span>
          <button onClick={() => {
            const url = prompt("Enter local path or GitHub URL to import project:");
            if(url) alert("Project imported: " + url + "\n(Agent context updated)");
          }} className="text-textMuted hover:text-textMain transition-colors" title="Import Project">
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="px-3 py-2 border-b border-surfaceBorder">
          <div className="text-xs text-textMuted p-2 hover:bg-surface rounded cursor-pointer border border-transparent hover:border-surfaceBorder transition-colors flex items-center gap-2">
            <Database className="w-3 h-3" /> Default Workspace
          </div>
        </div>

        <div className="px-4 py-3 flex items-center justify-between border-b border-surfaceBorder">
          <span className="text-[10px] font-bold tracking-[0.2em] text-white uppercase">Sessions</span>
          <button onClick={createNewSession} className="text-textMuted hover:text-textMain transition-colors">
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
                title={`Updated ${new Date(s.updatedAt).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: 'numeric', hour12: true })}`}
                className={`group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all text-xs font-mono min-w-0 ${
                  s.id === currentSessionId
                    ? 'bg-surface text-textMain'
                    : 'text-textMuted hover:text-textMuted hover:bg-surface'
                }`}
              >
                <span className="truncate flex-1 min-w-0 block">{s.title}</span>
                <button onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity ml-1 text-textMuted hover:text-rose-400">
                  <Trash2 className="w-3 h-3" />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Chat panel (now in middle) ───────────────────────────────────────────────── */}
      <div className="flex-1 min-w-[400px] flex flex-col bg-surface border-r border-surfaceBorder">

        {/* Chat header + model selector */}
        <div className="h-12 border-b border-surfaceBorder flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.8)]" />
            <span className="text-[10px] font-bold tracking-[0.2em] text-textMuted uppercase">Cipher Chat</span>
          </div>

          {/* Model dropdown */}
          <div className="relative">
            <button onClick={() => setShowModelDropdown(v => !v)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-surfaceBorder bg-surface hover:bg-surface transition-colors text-xs font-mono">
              <activeModel.icon className="w-3.5 h-3.5" />
              <span className="text-textMuted">{activeModel.name}</span>
              <ChevronDown className="w-3 h-3 text-textMuted" />
            </button>
            <AnimatePresence>
              {showModelDropdown && (
                <motion.div initial={{ opacity: 0, y: 6, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 6, scale: 0.97 }}
                  className="absolute right-0 top-9 w-64 rounded-xl border border-surfaceBorder bg-surface shadow-2xl overflow-hidden z-50 py-1.5">
                  {MODELS.map(m => (
                    <div key={m.id} onClick={() => { setActiveModelId(m.id); setShowModelDropdown(false); }}
                      className="flex items-center gap-3 px-4 py-2.5 hover:bg-surface cursor-pointer transition-colors">
                      <m.icon className="w-3.5 h-3.5 text-textMuted" />
                      <span className="text-xs font-mono text-textMuted">{m.name}</span>
                      {activeModelId === m.id && <span className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-500" />}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scrollbar-hide px-4 py-4 space-y-4">
          <AnimatePresence initial={false}>
            {currentSession?.messages.map(msg => (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`max-w-[88%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-accent text-white rounded-tr-sm font-medium'
                    : msg.role === 'system'
                    ? 'bg-rose-950/40 border border-rose-500/20 text-rose-300 rounded-tl-sm font-mono text-xs'
                    : 'bg-surface border border-surfaceBorder text-white rounded-tl-sm'
                }`}>
                  {msg.content}
                  
                  {/* Display Attachments */}
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {msg.attachments.map((file, i) => (
                        <div key={i} className="flex flex-col gap-1 max-w-[200px]">
                          {file.content.startsWith('data:image/') ? (
                            <img src={file.content} alt={file.name} className={`w-full rounded-lg border object-cover max-h-[150px] ${msg.role === 'user' ? 'border-gray-300' : 'border-surfaceBorder'}`} />
                          ) : (
                            <div className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border ${msg.role === 'user' ? 'bg-gray-100 border-gray-200' : 'bg-surface border-surfaceBorder'}`}>
                              <FileText className={`w-4 h-4 ${msg.role === 'user' ? 'text-gray-500' : 'text-textMuted'}`} />
                              <span className={`truncate text-xs ${msg.role === 'user' ? 'text-gray-700' : 'text-textMuted'}`}>{file.name}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Agent result cards */}
                {msg.agents_used && msg.agents_used.length > 0 && (
                  <div className="mt-3 w-full space-y-2 max-w-[96%]">
                    <div className="text-[10px] text-textMuted font-mono uppercase tracking-wider flex items-center gap-2">
                      <span className="flex-1 h-px bg-surface" /> agents executed <span className="flex-1 h-px bg-surface" />
                    </div>
                    {msg.results?.map((res, idx) => {
                      const agentId = msg.agents_used![idx] || 'ai_specialist';
                      const meta = getAgentMeta(agentId);
                      const Icon = meta.icon;
                      return (
                        <motion.div key={idx} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.08 }}
                          className="p-3 rounded-xl border text-xs font-mono"
                          style={{ borderColor: meta.color, backgroundColor: meta.glow, color: 'var(--textMain)' }}>
                          <div className="flex items-center gap-2 mb-2 font-bold uppercase tracking-wider" style={{ color: meta.color }}>
                            <Icon className="w-3 h-3" />{meta.name}
                          </div>
                          {res.error
                            ? <div className="text-rose-400 break-words">{res.error}</div>
                            : <div className="space-y-3 opacity-90">
                                <div>{res.summary || 'Agent completed successfully.'}</div>
                                {res.result && (
                                  <pre className="bg-surface p-3 rounded-lg overflow-x-auto text-[10px] text-emerald-400 border border-surfaceBorder whitespace-pre-wrap break-words">
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

                {/* Action buttons (Copy, Like, Dislike) */}
                {msg.role !== 'user' && (
                  <div className="flex items-center gap-1 mt-1.5 ml-2 text-textMuted">
                    <button onClick={() => handleCopy(msg.id, msg.content)} className="hover:text-textMain transition-colors p-1.5 rounded border border-transparent hover:border-surfaceBorder hover:bg-surface cursor-pointer" title={copiedId === msg.id ? "Copied!" : "Copy"}>
                      {copiedId === msg.id ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                    <button className="hover:text-emerald-400 transition-colors p-1.5 rounded border border-transparent hover:border-emerald-900 hover:bg-emerald-950/30 cursor-pointer" title="Good response">
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button className="hover:text-rose-400 transition-colors p-1.5 rounded border border-transparent hover:border-rose-900 hover:bg-rose-950/30 cursor-pointer" title="Bad response">
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}
                
                {/* User message action buttons */}
                {msg.role === 'user' && (
                  <div className="flex items-center gap-1 mt-1.5 mr-2 text-textMuted">
                    <button onClick={() => handleCopy(msg.id, msg.content)} className="hover:text-textMain transition-colors p-1.5 rounded border border-transparent hover:border-surfaceBorder hover:bg-surface cursor-pointer" title={copiedId === msg.id ? "Copied!" : "Copy"}>
                      {copiedId === msg.id ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-2 px-4 py-3 w-fit rounded-2xl rounded-tl-sm bg-surface border border-surfaceBorder">
              <div className="flex items-center gap-2">
                {['bg-blue-500','bg-purple-500','bg-cyan-500'].map((c, i) => (
                  <span key={i} className={`w-1.5 h-1.5 rounded-full ${c} animate-bounce`}
                    style={{ animationDelay: `${i * 0.1}s`, boxShadow: `0 0 6px currentColor` }} />
                ))}
              </div>
              <span className="text-[10px] text-textMuted font-mono">Generating response... (This may take several minutes)</span>
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
                style={{ borderColor: t.color, backgroundColor: t.glow }}
                className="flex-shrink-0 text-left w-36 p-2.5 rounded-xl border transition-all group">
                <div className="text-base mb-1">{t.emoji}</div>
                <div className="text-[11px] font-semibold text-textMuted group-hover:text-textMain transition-colors" style={{ color: t.color }}>{t.title}</div>
              </motion.button>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="p-4 border-t border-surfaceBorder">
          {/* Attachments preview */}
          {attachments.length > 0 && (
            <div className="flex gap-2 mb-2 flex-wrap">
              {attachments.map((file, i) => (
                <div key={i} className="flex items-center gap-1.5 px-3 py-1.5 bg-surface rounded-lg text-xs text-textMain border border-surfaceBorder">
                  <FileText className="w-3 h-3 text-textMuted" />
                  <span className="max-w-[150px] truncate">{file.name}</span>
                  <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))} className="ml-1 text-textMuted hover:text-textMain">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
          
          <div className="flex items-end gap-2 border border-surfaceBorder rounded-2xl p-2 bg-surface focus-within:border-surfaceBorder focus-within:shadow-[0_0_20px_rgba(255,255,255,0.03)] transition-all relative">
            {isRecording && <div className="absolute right-12 bottom-12 w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div>}
            
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" multiple accept="image/*,text/*,application/json,text/markdown,.py,.js,.jsx,.ts,.tsx,.html,.css,.csv,.xlsx,.xls" />
            <button onClick={() => fileInputRef.current?.click()} className="p-2 text-textMuted hover:text-textMuted transition-colors" title="Attach files">
              <Paperclip className="w-5 h-5" />
            </button>
            
            <textarea value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder="Describe a task for the agents… (Shift+Enter for newline)"
              className="flex-1 bg-transparent outline-none text-sm !text-textMain placeholder-textMuted font-mono resize-none min-h-[40px] max-h-32 py-2.5 px-1"
              rows={1}
            />

            <div className="flex items-center gap-2 mb-0.5">
              <button 
                onClick={toggleRecording} 
                className={`flex items-center justify-center w-10 h-10 rounded-full transition-all ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600 text-textMain shadow-[0_0_15px_rgba(239,68,68,0.4)]' 
                    : 'bg-transparent text-textMuted hover:text-textMuted hover:bg-surface'
                }`}
                title={isRecording ? "Stop recording" : "Voice note"}
              >
                {isRecording ? (
                  <div className="flex items-center gap-[3px] justify-center h-full">
                    <div className="w-1 h-2 bg-textMain rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1 h-3.5 bg-textMain rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1 h-2 bg-textMain rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                ) : (
                  <Mic className="w-5 h-5" />
                )}
              </button>

              <button 
                onClick={() => handleSend()} 
                disabled={isLoading || !input.trim()}
                className="flex items-center justify-center w-10 h-10 rounded-full bg-surface text-textMuted hover:bg-surface hover:text-textMain disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>
          {!backendOnline && backendOnline !== null && (
            <p className="text-[10px] text-rose-400/70 font-mono mt-2 text-center">
              Backend offline — start the orchestrator: <code>uvicorn orchestrator:app --port 8000</code>
            </p>
          )}
        </div>
      </div>

      <div className="w-[450px] shrink-0 flex flex-col bg-surface">
        {/* Header */}
        <div className="h-12 border-b border-surfaceBorder flex items-center justify-between px-5 bg-surface">
          <span className="text-[10px] font-bold tracking-[0.25em] text-textMuted uppercase">Live Agent Canvas</span>
          {/* Backend status indicator */}
          <div className={`flex items-center gap-2 text-[10px] font-mono px-3 py-1 rounded-full border ${
            backendOnline === null ? 'border-surfaceBorder text-textMuted' :
            backendOnline ? 'border-emerald-500/30 text-emerald-400 bg-emerald-400/5' : 'border-rose-500/30 text-rose-400 bg-rose-400/5'
          }`}>
            {backendOnline === null ? <Loader2 className="w-3 h-3 animate-spin" /> :
             backendOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {backendOnline === null ? 'Checking…' : backendOnline ? 'Backend online' : 'Backend offline'}
          </div>
        </div>

        {/* Canvas Area */}
        {(Object.keys(agentStates).length > 0) && (
          <div className="flex-1 w-full border-t border-surfaceBorder relative bg-surface overflow-hidden">
            <CanvasApp />
          </div>
        )}
      </div>
    </div>
  );
}
