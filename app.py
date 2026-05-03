"""
Thai Text Normalizer — Unified LLM-powered pipeline.

Single processing engine:
  1. ThaiTextNormalizer (rule-based pre-processing)
  2. Gemini LLM (context-aware normalization, requires API key)

Set GEMINI_API_KEY environment variable or enter it in the sidebar.
"""

import io
import streamlit as st
from pythainlp.tokenize import word_tokenize
from gtts import gTTS

from normalizer.smart_normalizer import SmartNormalizer
from config import GEMINI_API_KEY

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Thai NLP | Smart Text Normalizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Kanit', sans-serif;
    background-color: #F0F2F6;
    color: #1A202C;
}
.main-title {
    font-weight: 700;
    font-size: 3rem;
    color: #1A202C;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.sub-title {
    color: #4A5568;
    font-size: 1.15rem;
    margin-bottom: 2rem;
    font-weight: 300;
}
.stTextArea textarea {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E0;
    border-radius: 12px;
    color: #1A202C;
    font-size: 1.1rem;
    padding: 1rem;
    transition: all 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: #4299E1;
    box-shadow: 0 0 0 2px rgba(66,153,225,0.2);
}
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #4299E1;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.2rem;
    box-shadow: 0 8px 16px rgba(0,0,0,0.06);
    transition: transform 0.2s ease;
}
.result-card:hover { transform: translateY(-2px); }
.result-header {
    color: #718096;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
}
.result-text {
    color: #1A202C;
    font-size: 1.6rem;
    font-weight: 500;
    line-height: 1.6;
}
.step-card {
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
}
.step-label {
    color: #718096;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.step-value {
    color: #2D3748;
    font-size: 1.1rem;
    margin-top: 0.3rem;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #3182CE, #2B6CB0);
    color: white;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 12px rgba(49,130,206,0.3);
    transition: all 0.3s ease;
    width: 100%;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(49,130,206,0.4);
}
.footer {
    text-align: center;
    color: #A0AEC0;
    font-size: 0.85rem;
    margin-top: 4rem;
    padding: 1.5rem;
    border-top: 1px solid #E2E8F0;
}
.badge-llm {
    display: inline-block;
    background: #EBF8FF;
    color: #2B6CB0;
    border: 1px solid #BEE3F8;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-rules {
    display: inline-block;
    background: #F0FFF4;
    color: #276749;
    border: 1px solid #C6F6D5;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR — API KEY
# ==========================================
if "history" not in st.session_state:
    st.session_state["history"] = []

st.sidebar.markdown("### Settings")

# API key: env var takes priority; user can override in sidebar
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value="",
    type="password",
    placeholder="Leave blank to use env var",
    help="Set GEMINI_API_KEY as environment variable, or paste your key here.",
)
active_key = api_key_input.strip() or GEMINI_API_KEY

if st.sidebar.button("Reset Cache", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "**Pipeline:**  \n"
    "1. Rule-based pre-processing  \n"
    "2. Gemini LLM normalization"
)

# ==========================================
# 4. INIT NORMALIZER (cached singleton)
# ==========================================
@st.cache_resource
def get_normalizer(key: str) -> SmartNormalizer:
    return SmartNormalizer(api_key=key or None)

normalizer = get_normalizer(active_key)

# ==========================================
# 5. HELPERS
# ==========================================
def generate_audio(text: str) -> bytes:
    tts = gTTS(text=text, lang="th", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

# ==========================================
# 6. MAIN UI
# ==========================================
st.markdown('<h1 class="main-title">Thai Text Normalizer</h1>', unsafe_allow_html=True)

mode_badge = (
    '<span class="badge-llm">Gemini AI</span>'
    if normalizer.has_llm
    else '<span class="badge-rules">Rule-based only (no API key)</span>'
)
st.markdown(
    f'<p class="sub-title">ระบบ AI แปลงข้อความภาษาไทยไม่เป็นทางการ → ภาษาไทยมาตรฐาน '
    f'{mode_badge}</p>',
    unsafe_allow_html=True,
)

if not normalizer.has_llm:
    st.warning(
        "ไม่พบ Gemini API Key — ใช้เฉพาะ Rule-based engine. "
        "ใส่ API Key ในแถบด้านซ้ายเพื่อเปิดใช้งาน Gemini AI."
    )

st.divider()

st.markdown("### Input Text")
user_input = st.text_area(
    "Enter noisy Thai text:",
    height=140,
    placeholder="เช่น  ตะเองทำรายอยู่คับ / อ้วนน หิวข้าวจางงงเบยยย / ขาeกระเป๋าวันนี้คับ",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    process_btn = st.button("Normalize", type="primary")

# ==========================================
# 7. PROCESS
# ==========================================
if process_btn:
    if not user_input.strip():
        st.warning("กรุณาพิมพ์ข้อความก่อนกดปุ่ม")
    else:
        with st.spinner("กำลังประมวลผล..."):
            # Phase 1: rules only
            preprocessed = normalizer._rules.normalize(user_input)
            # Phase 2: full pipeline (rules + LLM)
            final_text = normalizer.normalize(user_input)

        # Save history
        st.session_state["history"].insert(0, {
            "original": user_input,
            "preprocessed": preprocessed,
            "normalized": final_text,
        })

        audio_bytes = generate_audio(final_text)

        # --- Result card ---
        st.divider()
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">Normalized Output</div>
            <div class="result-text">{final_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # --- Audio ---
        st.write("")
        st.markdown("### Listen")
        st.audio(audio_bytes, format="audio/mp3")

        # --- Processing steps ---
        st.write("")
        st.markdown("### Processing Steps")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">1. Original Input</div>
                <div class="step-value">{user_input}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">2. Rule-based Pre-processing</div>
                <div class="step-value">{preprocessed}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            label = "3. Gemini LLM Output" if normalizer.has_llm else "3. Final Output (rules only)"
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">{label}</div>
                <div class="step-value">{final_text}</div>
            </div>""", unsafe_allow_html=True)

        # --- Token view ---
        with st.expander("Token breakdown"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Input tokens")
                src_tokens = word_tokenize(user_input, engine="newmm")
                st.code(" | ".join(src_tokens), language="plaintext")
            with col_b:
                st.caption("Output tokens")
                out_tokens = word_tokenize(final_text, engine="newmm")
                st.code(" | ".join(out_tokens), language="plaintext")

        st.caption(f"Cache size: {normalizer.cache_size()} entries")

# ==========================================
# 8. HISTORY
# ==========================================
if st.session_state["history"]:
    st.write("")
    st.markdown("### History")
    for item in st.session_state["history"][:10]:
        with st.expander(f'"{item["original"][:50]}"'):
            cols = st.columns(3)
            cols[0].markdown(f"**Original**  \n{item['original']}")
            cols[1].markdown(f"**Pre-processed**  \n{item['preprocessed']}")
            cols[2].markdown(f"**Final**  \n{item['normalized']}")

    if st.button("Clear History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

st.markdown("""
<div class="footer">
    Developed by Baitong | AI Student @ Faculty of Science and Technology,
    Huachiew Chalermprakiet University (HCU)
</div>
""", unsafe_allow_html=True)
