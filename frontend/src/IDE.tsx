import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Files, Search, GitBranch, TerminalSquare, Settings, Send, Paperclip, ChevronDown, Cpu } from 'lucide-react';

const MODELS = [
  { id: 'orchestrator', name: 'CIPHER Orchestrator', desc: 'Central routing intelligence' },
  { id: 'fullstack', name: 'AYEESHA | Fullstack', desc: 'React, Vite, Node expert' },
  { id: 'data_science', name: 'DEEPTHI | Data', desc: 'Python & Data Analysis' },
  { id: 'security', name: 'MAHIMA | Security', desc: 'Cybersec & Auditing' },
  { id: 'devops', name: 'LIKITHA | DevOps', desc: 'Docker & Cloud architecture' },
  { id: 'ai_specialist', name: 'AI Specialist', desc: 'LLM Fine-tuning' },
];

const IDE = () => {
  const [activeModel, setActiveModel] = useState(MODELS[0]);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string}[]>([
    { role: 'assistant', content: "I'm Cipher." }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    const userMsg = input;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, language: 'English' })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: '[SYS_ERR]: Connection to Cipher backend failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen w-full flex bg-[#050505] text-white overflow-hidden font-sans selection:bg-white selection:text-black">
      
      {/* 1. Left Sidebar: Activity Bar */}
      <div className="w-14 flex flex-col items-center py-4 border-r border-[#1a1a1a] bg-[#0a0a0a] z-20">
        <div className="space-y-6">
          <Files className="w-5 h-5 text-white cursor-pointer hover:opacity-80 transition-opacity" />
          <Search className="w-5 h-5 text-[#555] hover:text-white cursor-pointer transition-colors" />
          <GitBranch className="w-5 h-5 text-[#555] hover:text-white cursor-pointer transition-colors" />
          <TerminalSquare className="w-5 h-5 text-[#555] hover:text-white cursor-pointer transition-colors" />
        </div>
        <div className="mt-auto">
          <Settings className="w-5 h-5 text-[#555] hover:text-white cursor-pointer transition-colors" />
        </div>
      </div>

      {/* 2. Left Sidebar: Explorer */}
      <div className="w-64 border-r border-[#1a1a1a] bg-[#080808] flex flex-col z-10">
        <div className="p-4 flex items-center justify-between">
          <span className="text-xs font-bold tracking-[0.2em] text-[#888] uppercase">Explorer</span>
        </div>
        <div className="px-2 space-y-1 text-sm font-mono text-[#aaa]">
          <div className="px-3 py-1.5 hover:bg-[#1a1a1a] rounded-md cursor-pointer flex items-center space-x-3 transition-colors">
            <span className="text-[#555]">▶</span> <span>src</span>
          </div>
          <div className="px-3 py-1.5 hover:bg-[#1a1a1a] rounded-md cursor-pointer flex items-center space-x-3 transition-colors">
            <span className="text-[#555]">≡</span> <span className="text-white">App.tsx</span>
          </div>
          <div className="px-3 py-1.5 hover:bg-[#1a1a1a] rounded-md cursor-pointer flex items-center space-x-3 transition-colors">
            <span className="text-[#555]">#</span> <span>index.css</span>
          </div>
        </div>
      </div>

      {/* 3. Middle: Code Editor */}
      <div className="flex-1 flex flex-col bg-[#050505] relative border-r border-[#1a1a1a]">
        {/* Subtle grid background */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.03]" 
             style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)', backgroundSize: '40px 40px' }} 
        />
        
        {/* Version Change Module / Header */}
        <div className="h-14 border-b border-[#1a1a1a] flex items-center justify-between px-6 bg-[#080808] z-30">
          <div className="flex space-x-4 text-xs font-mono">
            <span className="text-white border-b border-white pb-1">App.tsx</span>
            <span className="text-[#555] hover:text-[#aaa] cursor-pointer">index.css</span>
          </div>

          {/* Dedicated Model Selector (Like Antigravity) */}
          <div className="relative">
            <button 
              onClick={() => setShowModelDropdown(!showModelDropdown)}
              className="flex items-center space-x-3 px-4 py-1.5 rounded-full border border-[#222] bg-[#0a0a0a] hover:bg-[#111] transition-colors"
            >
              <Cpu className="w-4 h-4 text-[#888]" />
              <span className="text-sm font-mono tracking-tight">{activeModel.name}</span>
              <ChevronDown className="w-4 h-4 text-[#555]" />
            </button>

            <AnimatePresence>
              {showModelDropdown && (
                <motion.div 
                  initial={{ opacity: 0, y: 10, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.98 }}
                  className="absolute right-0 top-10 w-72 glass-panel rounded-xl border border-[#222] shadow-2xl overflow-hidden py-2 z-50"
                >
                  <div className="px-4 py-2 text-xs font-mono text-[#555] uppercase tracking-wider border-b border-[#222] mb-2">Select Agent Module</div>
                  {MODELS.map(model => (
                    <div 
                      key={model.id}
                      onClick={() => { setActiveModel(model); setShowModelDropdown(false); }}
                      className="px-4 py-3 hover:bg-[#1a1a1a] cursor-pointer transition-colors group"
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-mono text-white group-hover:text-blue-400">{model.name}</span>
                        {activeModel.id === model.id && <span className="w-2 h-2 rounded-full bg-blue-500" />}
                      </div>
                      <div className="text-xs text-[#555] mt-1">{model.desc}</div>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="flex-1 p-6 overflow-auto text-sm font-mono text-[#ccc] z-10">
          <pre className="outline-none leading-relaxed">
{`function App() {
  // Cipher initialized successfully.
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/ide" element={<IDE />} />
      </Routes>
    </Router>
  );
}`}
          </pre>
        </div>
      </div>

      {/* 4. Right Sidebar: Chat Agent */}
      <div className="w-[450px] bg-[#080808] flex flex-col z-20">
        
        {/* Chat Header */}
        <div className="h-14 border-b border-[#1a1a1a] flex items-center px-6">
          <span className="text-xs font-bold tracking-[0.2em] text-white uppercase">Cipher Communication</span>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg, i) => (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={i} 
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[85%] p-4 rounded-xl text-sm leading-relaxed font-mono ${
                msg.role === 'user' 
                  ? 'bg-white text-black rounded-tr-none' 
                  : 'bg-[#111] border border-[#222] text-[#ccc] rounded-tl-none'
              }`}>
                {msg.content}
              </div>
            </motion.div>
          ))}
          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="bg-[#111] border border-[#222] p-4 rounded-xl rounded-tl-none flex space-x-2 items-center">
                <span className="w-1.5 h-1.5 bg-[#555] rounded-full animate-bounce" />
                <span className="w-1.5 h-1.5 bg-[#555] rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <span className="w-1.5 h-1.5 bg-[#555] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              </div>
            </motion.div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 bg-[#080808] border-t border-[#1a1a1a]">
          <div className="border border-[#333] rounded-xl p-2 flex items-center bg-[#0a0a0a] focus-within:border-[#666] transition-colors">
            <button className="p-2 text-[#555] hover:text-white transition-colors">
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Query Cipher..."
              className="flex-1 bg-transparent border-none outline-none px-3 text-white placeholder-[#444] text-sm font-mono"
            />
            <button 
              onClick={handleSend}
              className="p-2 bg-white rounded-lg text-black hover:bg-gray-200 transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default IDE;
