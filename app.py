import streamlit as st
import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.corpus import thai_words
from pythainlp.util import Trie
from datasets import load_dataset

# ==========================================
# 1. PAGE CONFIG & THEME
# ==========================================
st.set_page_config(
    page_title="Thai NLP | Text Normalizer Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS INJECTION
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #0E1117; color: #FAFAFA; }
    .main-title { font-weight: 700; font-size: 3rem; color: #FAFAFA; margin-bottom: 0.5rem; letter-spacing: -1px; }
    .sub-title { color: #888888; font-size: 1.1rem; margin-bottom: 2rem; }
    .stTextArea textarea { background-color: #161A21; border: 1px solid #2D343F; border-radius: 12px; color: #FAFAFA; font-size: 1rem; padding: 1rem; }
    .stTextArea textarea:focus { border-color: #FF4B4B; box-shadow: 0 0 0 1px #FF4B4B; }
    .result-card { background-color: #161A21; border: 1px solid #2D343F; border-radius: 16px; padding: 1.5rem; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .result-header { color: #888888; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .result-text { color: #FAFAFA; font-size: 1.5rem; font-weight: 600; line-height: 1.4; }
    .stCodeBlock { background-color: #1A1F27 !important; border-radius: 12px !important; border: 1px solid #2D343F !important; }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 12px; padding: 0.75rem 1.5rem; font-weight: 600; border: none; width: 100%; transition: all 0.2s ease; }
    .stButton>button:hover { background-color: #E64444; box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3); }
    .footer { text-align: center; color: #444444; font-size: 0.8rem; margin-top: 4rem; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD SLANG DICTIONARY FROM HUGGING FACE
# ==========================================
@st.cache_data
def load_slang_dict():
    try:
        # Load dataset directly from Hugging Face repository
        dataset = load_dataset("thassanawalai/thai-social-slang-dict", split="train")
        
        # Map slang and formal columns into a Python Dictionary
        return dict(zip(dataset['slang'], dataset['formal']))
        
    except Exception as e:
        st.error(f"Failed to fetch data from Hugging Face: {e}")
        return {}

slang_dict = load_slang_dict()

# Prepare custom tokenizer trie incorporating the slang dictionary
standard_words = set(thai_words())
custom_words = standard_words.copy()
if slang_dict:
    custom_words.update(slang_dict.keys())
custom_tokenizer_trie = Trie(custom_words)

# ==========================================
# 4. CORE PROCESSING FUNCTION
# ==========================================
@st.cache_data
def auto_normalize_text(text):
    # Reduce character elongation
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Tokenize using the custom trie dictionary
    tokens = word_tokenize(reduced_text, engine='newmm', custom_dict=custom_tokenizer_trie)
    
    smart_tokens = []
    for word in tokens:
        # Preserve whitespaces and non-Thai characters
        if not word.strip() or not re.match(r'^[ก-๙]+$', word):
            smart_tokens.append(word)
        # Apply slang mapping
        elif word in slang_dict:
            smart_tokens.append(slang_dict[word])
        # Skip spell check for valid standard Thai words
        elif word in standard_words:
            smart_tokens.append(word)
        # Apply spell correction for unknown words longer than 1 character
        elif len(word) > 1:
            smart_tokens.append(correct(word))
        else:
            smart_tokens.append(word)
            
    return "".join(smart_tokens), tokens, smart_tokens

# ==========================================
# 5. MAIN APPLICATION UI
# ==========================================
st.markdown('<h1 class="main-title">Thai NLP Text Normalizer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An intelligent system to transform informal Thai social media text and slang into proper, formal sentences.</p>', unsafe_allow_html=True)
st.divider()

st.markdown("### Input Text")
user_input = st.text_area(
    "Enter noisy Thai text below:", 
    height=150, 
    placeholder="e.g., อ้วนน หิวข้าวจางงงเบยยย...",
    label_visibility="collapsed"
)

col_btn, _ = st.columns([1, 2])
with col_btn:
    process_btn = st.button("Normalize & Correct", type="primary")

if process_btn:
    if user_input.strip():
        with st.spinner('AI Engine is processing...'):
            cleaned_text, raw_tokens, smart_tokens = auto_normalize_text(user_input)
        
        st.divider()
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">Normalized Output</div>
            <div class="result-text">{cleaned_text}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("##")
        
        st.markdown("### Under the Hood: Tokenization Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Raw Tokens (Before)")
            st.code(raw_tokens, language="python")
        with col2:
            st.markdown("#### Normalized Tokens (After)")
            st.code(smart_tokens, language="python")
    else:
        st.warning("Please enter some text before processing.")

st.markdown("""
<div class="footer">
    Developed by Baitong | AI Undergraduate @ Faculty of Science and Technology, Huachiew Chalermprakiet University (HCU)
</div>
""", unsafe_allow_html=True)