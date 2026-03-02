import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct

# ==========================================
# ✨ 1. PAGE CONFIG & THEME (Must be FIRST)
# ==========================================
st.set_page_config(
    page_title="Thai NLP | Text Normalizer Pro",
    page_icon="🇹🇭",
    layout="wide", # Use full width for a modern look
    initial_sidebar_state="collapsed" # Hide sidebar for focus
)

# ==========================================
# 🎨 2. CUSTOM CSS INJECTION (The Magic!)
# ==========================================
# We use st.markdown with unsafe_allow_html=True to inject custom styles.
st.markdown("""
<style>
    /* 🔠 Import Modern Google Font (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 🧱 Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; /* Dark Mode Background */
        color: #FAFAFA;
    }

    /* 🏠 Main Title Styling */
    .main-title {
        font-weight: 700;
        font-size: 3rem;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-title {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* 📝 Input Area Styling */
    .stTextArea textarea {
        background-color: #161A21;
        border: 1px solid #2D343F;
        border-radius: 12px;
        color: #FAFAFA;
        font-size: 1rem;
        padding: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #FF4B4B; /* Primary Color Accent */
        box-shadow: 0 0 0 1px #FF4B4B;
    }

    /* 🎯 Result Box Styling (Professional Card) */
    .result-card {
        background-color: #161A21;
        border: 1px solid #2D343F;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .result-header {
        color: #888888;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .result-text {
        color: #FAFAFA;
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.4;
    }

    /* 🔍 Analysis Boxes (Code Blocks) */
    .stCodeBlock {
        background-color: #1A1F27 !important;
        border-radius: 12px !important;
        border: 1px solid #2D343F !important;
    }

    /* 🔘 Primary Button Styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #E64444;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }

    /* 🧹 Footer */
    .footer {
        text-align: center;
        color: #444444;
        font-size: 0.8rem;
        margin-top: 4rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 3. CORE PROCESSING FUNCTION
# ==========================================
@st.cache_data
def auto_normalize_text(text):
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = word_tokenize(reduced_text, engine='newmm')
    smart_tokens = [correct(word) for word in tokens]
    return "".join(smart_tokens), tokens, smart_tokens

# ==========================================
# 🏠 4. MAIN APPLICATION UI
# ==========================================
# Title Section (Using custom HTML classes for styling)
st.markdown('<h1 class="main-title">Thai NLP Text Normalizer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An intelligent system to transform informal Thai social media text and slang into proper, formal sentences.</p>', unsafe_allow_html=True)

st.divider()

# Input Section
st.markdown("### 📝 Input Text")
user_input = st.text_area(
    "Enter noisy Thai text below:", 
    height=150, 
    placeholder="e.g., อ้วนน หิวข้าวจางงงเบยยย...",
    label_visibility="collapsed" # Hide label for a cleaner look
)

# Process Button
col_btn, _ = st.columns([1, 2]) # Place button on the left
with col_btn:
    process_btn = st.button("Normalize & Correct", type="primary")

# Results & Analysis Section
if process_btn:
    if user_input.strip():
        with st.spinner('AI Engine is processing...'):
            cleaned_text, raw_tokens, smart_tokens = auto_normalize_text(user_input)
        
        st.divider()
        
        # 🎯 Modern Result Card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">Normalized Output</div>
            <div class="result-text">{cleaned_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("##") # Add space
        
        # 🔍 Analysis Section (Columns)
        st.markdown("### 🔍 Under the Hood: Tokenization Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Raw Tokens (Before)")
            st.code(raw_tokens, language="python")
            
        with col2:
            st.markdown("#### 🟢 Normalized Tokens (After)")
            st.code(smart_tokens, language="python")
            
    else:
        st.warning("⚠️ Please enter some text before processing.")

# ==========================================
# 🧹 5. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    Developed by Baitong | AI Undergraduate @ Faculty of Science and Technology, Huachiew Chalermprakiet University (HCU)
</div>
""", unsafe_allow_html=True)