import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const Landing = () => {
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });
  const [loaded, setLoaded] = useState(false);

  // Smooth Loader simulation
  useEffect(() => {
    const t = setTimeout(() => setLoaded(true), 2500);
    return () => clearTimeout(t);
  }, []);

  // Interactive ASCII Matrix Effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*';
    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    const rows = Math.floor(canvas.height / fontSize);
    
    // Store grid characters
    const grid: string[][] = Array(columns).fill(0).map(() => Array(rows).fill(' '));
    for(let i=0; i<columns; i++) {
      for(let j=0; j<rows; j++) {
        if(Math.random() > 0.95) grid[i][j] = chars[Math.floor(Math.random() * chars.length)];
      }
    }

    let animationFrameId: number;

    const render = () => {
      ctx.fillStyle = 'rgba(5, 5, 5, 0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.font = `${fontSize}px monospace`;
      ctx.textAlign = 'center';

      for(let i=0; i<columns; i++) {
        for(let j=0; j<rows; j++) {
          const char = grid[i][j];
          if (char === ' ') continue;

          const x = i * fontSize;
          const y = j * fontSize;
          
          // Distance to mouse
          const dx = mousePos.x - x;
          const dy = mousePos.y - y;
          const dist = Math.sqrt(dx*dx + dy*dy);
          
          let opacity = 0.1;
          if (dist < 150) {
            opacity = 1 - (dist / 150);
            if(Math.random() > 0.8) grid[i][j] = chars[Math.floor(Math.random() * chars.length)];
          }

          ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
          ctx.fillText(char, x, y);
        }
      }
      animationFrameId = requestAnimationFrame(render);
    };
    render();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [mousePos]);

  return (
    <div 
      className="h-screen w-full bg-background relative overflow-hidden flex items-center justify-center cursor-crosshair"
      onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
    >
      <canvas ref={canvasRef} className="absolute inset-0 z-0 opacity-40 mix-blend-screen" />
      
      {/* Thermal Glow tracking mouse */}
      <motion.div 
        animate={{ x: mousePos.x - 200, y: mousePos.y - 200 }}
        transition={{ type: "tween", ease: "linear", duration: 0 }}
        className="absolute w-[400px] h-[400px] bg-white rounded-full opacity-5 blur-[100px] pointer-events-none z-0"
      />

      <AnimatePresence>
        {!loaded ? (
          <motion.div
            key="loader"
            exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
            transition={{ duration: 0.8 }}
            className="z-10 text-white font-mono text-sm tracking-[0.5em] uppercase"
          >
            <motion.span animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.5, repeat: Infinity }}>
              Decrypting Core...
            </motion.span>
          </motion.div>
        ) : (
          <motion.div 
            key="content"
            initial={{ opacity: 0, filter: "blur(20px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
            className="z-10 flex flex-col items-center"
          >
            <div className="overflow-hidden mix-blend-difference mb-8">
              <motion.h1 
                initial={{ y: "100%" }}
                animate={{ y: 0 }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="text-[120px] leading-none font-bold tracking-tighter text-white"
              >
                CIPHER.
              </motion.h1>
            </div>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate('/ide')}
              className="glow-border px-8 py-4 bg-transparent border border-white/20 text-white font-mono text-xs uppercase tracking-[0.2em] flex items-center space-x-4 hover:bg-white hover:text-black transition-all duration-300"
            >
              <span>Initialize Workspace</span>
              <span className="w-1.5 h-1.5 bg-current rounded-full animate-pulse" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Landing;
