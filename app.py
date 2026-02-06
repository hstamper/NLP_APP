import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from fastcoref import FCoref
import streamlit as st

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP By Hilliard",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

from utils.model_loader import hf_authenticate
from utils.task_registry import TASK_REGISTRY

# ─── Authentication & Runtime ────────────────────────────────────────────────────
hf_authenticate()
torch.set_num_threads(1)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- Import Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---------- Root Variables ---------- */
    :root {
        --primary: #B91C1C;
        --primary-light: #DC2626;
        --primary-bg: #FEF2F2;
        --cream: #FDF6EC;
        --cream-dark: #F5E6D0;
        --surface: #FFFDF9;
        --surface-alt: #FDF6EC;
        --border: #E8D5C0;
        --text-primary: #1C1412;
        --text-secondary: #5C4033;
        --text-muted: #9C8575;
        --success: #059669;
        --success-bg: #ECFDF5;
        --warning: #D97706;
        --warning-bg: #FFFBEB;
        --radius: 12px;
        --shadow-sm: 0 1px 2px rgba(28,20,18,0.04);
        --shadow-md: 0 4px 12px rgba(28,20,18,0.06);
        --shadow-lg: 0 8px 30px rgba(28,20,18,0.08);
    }

    /* ---------- Global Overrides ---------- */
    .stApp {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--cream);
    }

    /* Header area */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- Hero Section ---------- */
    .hero-container {
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 40%, #DC2626 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero-logo {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        color: rgba(253,246,236,0.85);
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* ---------- Task Badge ---------- */
    .task-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--primary-bg);
        color: var(--primary);
        font-weight: 600;
        font-size: 0.8rem;
        padding: 6px 14px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* ---------- Section Cards ---------- */
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    .section-card:hover {
        box-shadow: var(--shadow-md);
    }
    .section-card h3 {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 1rem 0;
    }

    /* ---------- Result Card ---------- */
    .result-card {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-card h4 {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 0.75rem 0;
    }

    /* ---------- Result Items ---------- */
    .result-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 10px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-item:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-sm);
    }
    .result-index {
        background: var(--primary-bg);
        color: var(--primary);
        font-weight: 600;
        font-size: 0.75rem;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    /* ---------- NER Entity Tag ---------- */
    .entity-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.25rem 0.25rem;
        font-size: 0.9rem;
    }
    .entity-label {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--text-primary);
    }
    .entity-type {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .entity-type-per { background: #DBEAFE; color: #1D4ED8; }
    .entity-type-loc { background: #D1FAE5; color: #065F46; }
    .entity-type-org { background: #FEF3C7; color: #92400E; }
    .entity-type-misc { background: #F3E8FF; color: #6B21A8; }
    .entity-type-default { background: #F1F5F9; color: #475569; }

    /* ---------- Model Info Pill ---------- */
    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    .model-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary);
        flex-shrink: 0;
    }

    /* ---------- Sidebar Styling ---------- */
    section[data-testid="stSidebar"] {
        background: #FAF5ED;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    /* ---------- Button Styling ---------- */
    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, #B91C1C, #DC2626);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(185,28,28,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(185,28,28,0.4);
    }

    /* ---------- Input Styling ---------- */
    .stTextInput input, .stTextArea textarea {
        font-family: 'DM Sans', sans-serif;
        border-radius: 10px;
        border: 1.5px solid var(--border);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary-light);
        box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
    }

    /* ---------- Spinner ---------- */
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
    .footer p {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ---------- Divider ---------- */
    .section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    /* Hide default Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <span class="hero-logo">🔬</span>
    <p class="hero-title">Hilliard's Professional NLP</p>
    <p class="hero-subtitle">Explore state-of-the-art natural language processing models interactively. Select a task, provide input, and see results in real time.</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")

    # Task selection
    display_name_to_key = {cfg["name"]: key for key, cfg in TASK_REGISTRY.items()}
    selected_task_name = st.selectbox(
        "NLP Task",
        list(display_name_to_key.keys()),
        help="Choose the NLP task you want to explore."
    )
    task_key = display_name_to_key[selected_task_name]
    task_config = TASK_REGISTRY[task_key]

    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")

    use_custom_model = st.toggle("Use custom model", value=False)
    if use_custom_model:
        model_name = st.text_input(
            "Hugging Face Model ID",
            value=task_config["default_model"],
            help="Enter a valid Hugging Face model identifier."
        )
    else:
        model_name = task_config["default_model"]

    st.markdown(f"""
    <div class="model-pill">
        <span class="model-pill-dot"></span>
        {model_name}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem; color:#94A3B8;">Models are loaded from '
        '<a href="https://huggingface.co" target="_blank" style="color:#DC2626;">Hugging Face</a>. '
        'First run may take a moment to download.</p>',
        unsafe_allow_html=True,
    )


# ─── Helper Functions ────────────────────────────────────────────────────────────

def render_result_header(label):
    """Render a styled result section header."""
    st.markdown(f"""
    <div class="result-card">
        <h4>{label}</h4>
    </div>
    """, unsafe_allow_html=True)


def render_list_results(items, label="Results"):
    """Render a list of results as styled items."""
    html = f'<div class="result-card"><h4>{label}</h4>'
    for i, item in enumerate(items, 1):
        display = str(item)
        html += f"""
        <div class="result-item">
            <span class="result-index">{i}</span>
            <span>{display}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_ner_results(entities):
    """Render NER results as styled entity tags."""
    html = '<div class="result-card"><h4>Extracted Entities</h4><div style="display:flex;flex-wrap:wrap;gap:4px;">'
    type_class_map = {
        "PER": "entity-type-per",
        "LOC": "entity-type-loc",
        "ORG": "entity-type-org",
        "MISC": "entity-type-misc",
    }
    for ent in entities:
        word = ent.get("word", "")
        etype = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
        css_class = type_class_map.get(etype, "entity-type-default")
        html += f"""
        <div class="entity-tag">
            <span class="entity-label">{word}</span>
            <span class="entity-type {css_class}">{etype}</span>
        </div>"""
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sentiment_results(results):
    """Render sentiment analysis results."""
    html = '<div class="result-card"><h4>Sentiment Analysis</h4>'
    for res in results:
        label = res.get("label", "UNKNOWN")
        score = res.get("score", 0)
        pct = f"{score * 100:.1f}%"
        emoji = "🟢" if label == "POSITIVE" else "🔴" if label == "NEGATIVE" else "🟡"
        bar_color = "#059669" if label == "POSITIVE" else "#B91C1C" if label == "NEGATIVE" else "#D97706"
        html += f"""
        <div style="margin-bottom:0.75rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                <span style="font-family:'DM Sans';font-weight:600;font-size:0.95rem;color:#1C1412;">
                    {emoji} {label}
                </span>
                <span style="font-family:'JetBrains Mono';font-size:0.85rem;color:#5C4033;">
                    {pct}
                </span>
            </div>
            <div style="background:#E8D5C0;border-radius:6px;height:8px;overflow:hidden;">
                <div style="background:{bar_color};height:100%;width:{score*100}%;border-radius:6px;transition:width 0.4s ease;"></div>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_text_result(text, label="Result"):
    """Render a single text result."""
    st.markdown(f"""
    <div class="result-card">
        <h4>{label}</h4>
        <p style="font-family:'DM Sans';font-size:1rem;color:#1C1412;line-height:1.7;margin:0;">{text}</p>
    </div>
    """, unsafe_allow_html=True)


def render_dict_result(data, label="Result"):
    """Render a dict result."""
    html = f'<div class="result-card"><h4>{label}</h4>'
    for k, v in data.items():
        html += f"""
        <div class="result-item">
            <span style="font-weight:600;color:#5C4033;min-width:100px;">{k}</span>
            <span>{v}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─── Active Task Badge ──────────────────────────────────────────────────────────
st.markdown(f'<div class="task-badge">▶ {selected_task_name}</div>', unsafe_allow_html=True)


# ─── Task Execution ──────────────────────────────────────────────────────────────
input_type = task_config["input_type"]
func = task_config["function"]
default_input = task_config["default_input"]
button_label = task_config.get("button_label", "Run")
doc_label = task_config.get("doc", "Output")

if input_type == "TEXT":
    st.markdown('<div class="section-card"><h3>Input</h3>', unsafe_allow_html=True)
    if len(str(default_input)) > 50:
        text = st.text_area("Enter your text", default_input, height=120, label_visibility="collapsed")
    else:
        text = st.text_input("Enter your text", default_input, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"⚡ {button_label}", use_container_width=True):
        with st.spinner("Processing..."):
            result = func(text, model_name=model_name)

        # Smart result rendering
        if task_key == "ner" and isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            render_ner_results(result)
        elif task_key == "sentiment" and isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            render_sentiment_results(result)
        elif isinstance(result, list):
            render_list_results(result, label=doc_label)
        elif isinstance(result, dict):
            render_dict_result(result, label=doc_label)
        else:
            render_text_result(str(result), label=doc_label)

elif input_type == "QA":
    st.markdown('<div class="section-card"><h3>Context</h3>', unsafe_allow_html=True)
    context = st.text_area("Provide context", default_input["context"], height=120, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Question</h3>', unsafe_allow_html=True)
    question = st.text_input("Ask a question", default_input["question"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"⚡ {button_label}", use_container_width=True):
        with st.spinner("Finding answer..."):
            answer = func(question, context, model_name=model_name)
        render_text_result(answer, label="Answer")

elif input_type == "TRANSLATION":
    st.markdown('<div class="section-card"><h3>Input Text</h3>', unsafe_allow_html=True)
    text = st.text_area("Text to translate", default_input, height=100, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.text_input("Source Language", task_config["params"]["src_lang"])
    with col2:
        tgt_lang = st.text_input("Target Language", task_config["params"]["tgt_lang"])

    if st.button(f"⚡ {button_label}", use_container_width=True):
        with st.spinner("Translating..."):
            result = func(text, src_lang=src_lang, tgt_lang=tgt_lang, model_name=model_name)
        render_text_result(result, label="Translation")

elif input_type == "STYLE":
    st.markdown('<div class="section-card"><h3>Input Text</h3>', unsafe_allow_html=True)
    text = st.text_area("Text to restyle", default_input, height=100, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    style_type = st.text_input("Target Style", task_config["params"]["style"])

    if st.button(f"⚡ {button_label}", use_container_width=True):
        with st.spinner("Applying style..."):
            result = func(text, style=style_type, model_name=model_name)
        render_text_result(result, label="Styled Text")

elif input_type == "TEXT_TO_TUPLE":
    st.markdown('<div class="section-card"><h3>Input Text</h3>', unsafe_allow_html=True)
    text = st.text_area("Text for masking demo", default_input, height=100, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f"⚡ {button_label}", use_container_width=True):
        with st.spinner("Generating masking demo..."):
            res1, res2 = func(text, model_name=model_name)

        st.markdown("""
        <div class="result-card">
            <h4>Masked Input IDs</h4>
        </div>
        """, unsafe_allow_html=True)
        st.code(res1, language="python")

        st.markdown("""
        <div class="result-card">
            <h4>Labels (Target Tokens)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.code(res2, language="python")


# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>NLP Playground · Powered by Hugging Face Transformers</p>
</div>
""", unsafe_allow_html=True)


