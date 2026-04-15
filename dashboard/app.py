import streamlit as st
import requests

st.set_page_config(
    page_title="CIPHER Multi-Agent Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .hero-sub {
        text-align: center;
        color: #8892a4;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #1e2433, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .template-btn {
        background: linear-gradient(135deg, #1e2433, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 0.8rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .result-box {
        background: #1e2433;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-badge {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── HERO SECTION ──
st.markdown('<div class="hero-title">⚡ CIPHER Multi-Agent Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Autonomous AI Platform — 4 Specialist Agents Working Together</div>', unsafe_allow_html=True)

# ── AGENT STATUS CARDS ──
st.markdown("### 🤖 Agent Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""<div class="agent-card">
        <div style="font-size:2rem">📊</div>
        <div style="color:#00d4ff;font-weight:700">Deepthi</div>
        <div style="color:#8892a4;font-size:0.8rem">Data Science</div>
        <div style="color:#00ff88;font-size:0.8rem">● Port 8001</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="agent-card">
        <div style="font-size:2rem">💻</div>
        <div style="color:#7b2ff7;font-weight:700">Ayeesha</div>
        <div style="color:#8892a4;font-size:0.8rem">Full Stack Dev</div>
        <div style="color:#00ff88;font-size:0.8rem">● Port 8002</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="agent-card">
        <div style="font-size:2rem">🔐</div>
        <div style="color:#ff6b6b;font-weight:700">Mahima</div>
        <div style="color:#8892a4;font-size:0.8rem">Cybersecurity</div>
        <div style="color:#00ff88;font-size:0.8rem">● Port 8003</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""<div class="agent-card">
        <div style="font-size:2rem">🚀</div>
        <div style="color:#ffd700;font-weight:700">Likitha</div>
        <div style="color:#8892a4;font-size:0.8rem">DevOps & Cloud</div>
        <div style="color:#00ff88;font-size:0.8rem">● Port 8004</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── TEMPLATES ──
st.markdown("### ⚡ Quick Templates")
st.markdown("<p style='color:#8892a4'>Click a template to auto-fill your task</p>", unsafe_allow_html=True)

templates = {
    "🔐 Login Page":     "Build a complete login page with email and password authentication, JWT tokens, and a registration form. Include frontend, backend API, and database schema.",
    "📝 Todo App":       "Build a full stack todo app where users can add, edit, delete and mark tasks as complete. Store tasks in a database with user accounts.",
    "🛒 E-commerce":     "Build an e-commerce product listing page with a shopping cart, product search, and checkout form. Include frontend and backend API.",
    "📊 Admin Dashboard":"Build an admin dashboard with charts showing user stats, recent activity table, and sidebar navigation. Include a FastAPI backend.",
    "📬 Contact Form":   "Build a contact form with name, email, subject and message fields. Include email validation, backend API to handle submissions, and a success message.",
    "👤 User Profile":   "Build a user profile page where users can view and edit their name, email, bio and profile picture. Include backend API and database schema.",
}

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

if 'task_input' not in st.session_state:
    st.session_state.task_input = ""

for i, (name, desc) in enumerate(templates.items()):
    with cols[i % 3]:
        if st.button(name, use_container_width=True, key=f"template_{i}"):
            st.session_state.task_input = desc

st.divider()

# ── TASK INPUT ──
st.markdown("### 🎯 Your Task")
task = st.text_area(
    "Describe what you want to build:",
    value=st.session_state.task_input,
    height=120,
    placeholder="e.g. Build a login page with email and password...",
    key="task_area"
)

col_run, col_clear = st.columns([4, 1])
with col_run:
    run_btn = st.button("🚀 Run All Agents", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.task_input = ""
        st.rerun()

# ── RUN ──
if run_btn:
    if not task:
        st.warning("⚠️ Please enter a task or pick a template first!")
    else:
        with st.spinner("🤖 Agents are working... this may take 1-3 minutes"):
            try:
                r = requests.post('http://localhost:8000/run',
                                  json={'description': task, 'context': ''})
                data = r.json()

                st.success("✅ All agents completed!")
                st.divider()

                agents_used = data.get('agents_used', [])
                st.markdown(f'<span class="status-badge">Agents used: {", ".join(agents_used)}</span>',
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                agent_icons = {
                    "fullstack": "💻", "data_science": "📊",
                    "security": "🔐", "devops": "🚀"
                }

                for i, result in enumerate(data.get('results', [])):
                    agent_name = agents_used[i] if i < len(agents_used) else f"Agent {i+1}"
                    icon = agent_icons.get(agent_name, "🤖")

                    st.markdown(f"### {icon} {agent_name.upper()} Agent")

                    if 'error' in result:
                        st.error(f"⚠️ {result['error']}")
                    else:
                        st.markdown(f"**📋 Summary:** {result.get('summary', '')}")
                        st.code(result.get('result', ''), language='python')
                        if result.get('next_agent'):
                            st.info(f"➡️ Passing to: {result.get('next_agent')} agent next")
                    st.divider()

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
                st.info("💡 Make sure the orchestrator is running on port 8000!")