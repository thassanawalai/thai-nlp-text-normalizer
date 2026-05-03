import streamlit as st
import re
import io
from pythainlp.corpus import thai_words
from pythainlp.util import Trie, normalize as thai_normalize
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from gtts import gTTS
from google import genai

from normalizer import ThaiTextNormalizer
from config import GEMINI_API_KEY

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Thai NLP | Text Normalizer Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    font-size: 3.5rem;
    color: #1A202C;
    margin-bottom: 0.5rem;
    letter-spacing: -1.5px;
}
.sub-title {
    color: #4A5568;
    font-size: 1.25rem;
    margin-top: 0.5rem;
    margin-bottom: 2.5rem;
    font-weight: 300;
}
.stTextArea textarea {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E0;
    border-radius: 12px;
    color: #1A202C;
    font-size: 1.15rem;
    padding: 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.stTextArea textarea:focus {
    border-color: #4299E1;
    box-shadow: 0 0 0 2px rgba(66, 153, 225, 0.2);
}
.result-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #4299E1;
    border-radius: 16px;
    padding: 2.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
}
.result-header {
    color: #718096;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
}
.result-text {
    color: #1A202C;
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1.6;
}
.stCodeBlock {
    background-color: #F7FAFC !important;
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #3182CE 0%, #2B6CB0 100%);
    color: white;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
    transition: all 0.3s ease;
    width: 100%;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(49, 130, 206, 0.4);
}
.footer {
    text-align: center;
    color: #A0AEC0;
    font-size: 0.9rem;
    margin-top: 5rem;
    padding: 1.5rem;
    border-top: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INITIALISE NORMALIZER AND TOKENIZER
# ==========================================
@st.cache_resource
def get_normalizer() -> ThaiTextNormalizer:
    """Load ThaiTextNormalizer once and cache for the session lifetime."""
    return ThaiTextNormalizer()

@st.cache_resource
def get_custom_tokenizer():
    """Build a custom tokenizer trie that includes slang words."""
    normalizer = get_normalizer()
    custom_words = set(thai_words())
    custom_words.update(normalizer._word_dict.keys())
    return Trie(custom_words)

normalizer = get_normalizer()
custom_trie = get_custom_tokenizer()

# ==========================================
# 4. CORE PROCESSING FUNCTION
# ==========================================
def auto_normalize_text(text: str) -> dict:
    """
    Normalize Thai text through a multi-step pipeline.
    Returns a dict with intermediate results for UI display.
    """
    steps = {}

    # Step 1: ThaiTextNormalizer (separator removal, char-sub, word mapping, elongation)
    normalized = normalizer.normalize(text)
    steps["step1_normalizer"] = normalized

    # Step 2: PyThaiNLP Unicode normalization (composite char cleanup)
    unicode_normalized = thai_normalize(normalized)
    steps["step2_unicode"] = unicode_normalized

    # Step 3: Tokenize with custom dictionary that knows slang terms
    tokens = word_tokenize(unicode_normalized, engine='newmm', custom_dict=custom_trie)
    steps["step3_tokenized"] = tokens

    # Step 4: Spell correction on unknown Thai tokens
    standard = set(thai_words())
    final_tokens = []
    changes_log = []
    for word in tokens:
        if not word.strip() or not re.match(r'^[ก-๙]+$', word):
            final_tokens.append(word)
        elif word in standard:
            final_tokens.append(word)
        elif len(word) > 1:
            corrected = correct(word)
            final_tokens.append(corrected)
            if word != corrected:
                changes_log.append({"word": word, "new_word": corrected, "action": "แก้คำผิด"})
        else:
            final_tokens.append(word)

    steps["step4_tokens"] = final_tokens
    steps["changes_log"] = changes_log
    steps["final_text"] = "".join(final_tokens)
    return steps


def generate_audio(text: str) -> bytes:
    tts = gTTS(text=text, lang='th', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

# ==========================================
# 5. SIDEBAR
# ==========================================
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.sidebar.markdown("### Settings")
if st.sidebar.button("Reset Cache", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

engine_choice = st.sidebar.radio(
    "Processing Engine:",
    ["Smart AI (Google Gemini)", "Rule-based (PyThaiNLP)"],
    help="AI understands context; rule-based is faster and offline."
)

# API key: use environment variable first, let user override in sidebar
api_key = GEMINI_API_KEY
if engine_choice == "Smart AI (Google Gemini)":
    entered = st.sidebar.text_input("Gemini API Key (optional override):", type="password")
    if entered:
        api_key = entered
    if not api_key:
        st.sidebar.warning("Set GEMINI_API_KEY env var or enter a key above.")
    st.sidebar.divider()

# ==========================================
# 6. MAIN UI
# ==========================================
st.markdown('<h1 class="main-title">Thai Text Normalizer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">ระบบประมวลผลและแปลงข้อความภาษาวัยรุ่น สแลง หรือคำวิบัติ '
    'ให้เป็นภาษาไทยที่ถูกต้องและเป็นทางการ</p>',
    unsafe_allow_html=True,
)
st.divider()

st.markdown("### Input Text")
user_input = st.text_area(
    "Enter noisy Thai text below:",
    height=150,
    placeholder="เช่น รับvายของ, ข า ย รถ, ขาeกระเป๋า, สวัสดีค้าบบบบ",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 2])
with col_btn:
    process_btn = st.button("Normalize & Correct", type="primary")

if process_btn:
    if not user_input.strip():
        st.warning("Please enter some text before processing.")
    elif engine_choice == "Smart AI (Google Gemini)":
        if not api_key:
            st.error("Please provide a Gemini API Key.")
        else:
            with st.spinner('AI is analyzing...'):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        "You are a linguistic expert in the Thai language. "
                        "Convert the following informal Thai social media text into a formal, "
                        "grammatically correct Thai sentence. Preserve the core meaning. "
                        "Output ONLY the formal Thai sentence.\n\n"
                        f"Informal Text: {user_input}\nFormal Text:"
                    )
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    cleaned_text = response.text.strip()

                    st.session_state['history'].insert(0, {
                        "original": user_input,
                        "normalized": cleaned_text,
                        "engine": "AI",
                    })

                    audio_bytes = generate_audio(cleaned_text)
                    analysis = auto_normalize_text(user_input)
                    ai_tokens = word_tokenize(cleaned_text, engine='newmm')

                    st.divider()
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">Normalized Output (Gemini AI)</div>
                        <div class="result-text">{cleaned_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("##")
                    st.markdown("### Listen to the Result")
                    st.audio(audio_bytes, format="audio/mp3")

                    st.write("##")
                    st.markdown("### Processing Steps")
                    with st.container(border=True):
                        st.markdown("##### 1. Text Normalization (ThaiTextNormalizer)")
                        col_a, col_b = st.columns(2)
                        col_a.info(f"**Input:** {user_input}")
                        col_b.success(f"**After normalizer:** {analysis['step1_normalizer']}")
                        st.divider()

                        st.markdown("##### 2. Source Tokenization")
                        st.code(" | ".join(analysis['step3_tokenized']), language="plaintext")
                        st.divider()

                        st.markdown("##### 3. AI Formalization Output")
                        st.code(" | ".join(ai_tokens), language="plaintext")

                except Exception as e:
                    st.error(f"API Error: {str(e)}")
    else:
        with st.spinner('Processing...'):
            results = auto_normalize_text(user_input)
            cleaned_text = results["final_text"]

            st.session_state['history'].insert(0, {
                "original": user_input,
                "normalized": cleaned_text,
                "engine": "Rule-Based",
            })

            audio_bytes = generate_audio(cleaned_text)

        st.divider()
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">Normalized Output</div>
            <div class="result-text">{cleaned_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("##")
        st.markdown("### Listen to the Result")
        st.audio(audio_bytes, format="audio/mp3")

        st.write("##")
        st.markdown("### Processing Steps")
        with st.container(border=True):
            st.markdown("##### 1. ThaiTextNormalizer")
            col_a, col_b = st.columns(2)
            col_a.info(f"**Input:** {user_input}")
            col_b.success(f"**After normalizer:** {results['step1_normalizer']}")
            st.divider()

            st.markdown("##### 2. Tokenization")
            st.code(" | ".join(results['step3_tokenized']), language="plaintext")
            st.divider()

            st.markdown("##### 3. Transformation Log")
            if results['changes_log']:
                for log in results['changes_log']:
                    st.markdown(f"- `{log['word']}` -> `{log['new_word']}` ({log['action']})")
            else:
                st.markdown("- No further corrections needed.")

# ==========================================
# 7. HISTORY
# ==========================================
if st.session_state['history']:
    st.write("##")
    st.markdown("### Recent History")
    for item in st.session_state['history'][:10]:
        with st.expander(f"[{item['engine']}] \"{item['original'][:40]}...\""):
            st.markdown(f"**Noisy:** {item['original']}")
            st.markdown(f"**Formal:** {item['normalized']}")

    if st.button("Clear History", use_container_width=True):
        st.session_state['history'] = []
        st.rerun()

st.markdown("""
<div class="footer">
    Developed by Baitong | AI Student @ Faculty of Science and Technology, Huachiew Chalermprakiet University (HCU)
</div>
""", unsafe_allow_html=True)
