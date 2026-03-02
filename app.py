import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct

# --- 1. Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Thai Text Normalizer",
    page_icon="🇹🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Core Processing Function ---
@st.cache_data # ทริคของเดฟ: ใส่ cache ให้ระบบจำผลลัพธ์เดิมเพื่อความรวดเร็ว
def auto_normalize_text(text):
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = word_tokenize(reduced_text, engine='newmm')
    smart_tokens = [correct(word) for word in tokens]
    return "".join(smart_tokens), tokens, smart_tokens

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ NLP Engine")
    st.markdown("""
    **Thai Text Normalizer System**
    
    This tool processes noisy Thai text by resolving:
    - **Word Elongation:** Shrinking repeated characters.
    - **Misspellings & Slang:** Correcting words using the Thai National Corpus (TNC).
    """)
    st.divider()
    st.caption("Developed by Baitong | AI Undergraduate @ HCU")

# --- 4. Main Application UI ---
st.title("✨ Thai Text Normalizer")
st.markdown("Transform informal Thai social media text and slang into proper, formal sentences.")

# Input Section
st.subheader("📝 Input Text")
user_input = st.text_area(
    "Enter noisy Thai text below:", 
    height=120, 
    placeholder="e.g., อ้วนน หิวข้าวจางงงเบยยย..."
)

# Process Button (ปุ่มใหญ่เต็มพื้นที่ ดูล้ำๆ)
if st.button("Normalize Text", type="primary", use_container_width=True):
    if user_input.strip():
        # แสดงแถบโหลดหมุนๆ ตอนที่ AI กำลังคิด
        with st.spinner('Normalizing text... Please wait.'):
            cleaned_text, raw_tokens, smart_tokens = auto_normalize_text(user_input)
        
        st.divider()
        
        # Results Section
        st.subheader("🎯 Result")
        st.success(f"**{cleaned_text}**") # โชว์ข้อความที่แก้แล้วเด่นๆ
        
        # Analysis Section (แบ่งคอลัมน์ซ้าย-ขวา)
        st.write("### 🔍 Tokenization Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Before (Raw Tokens)")
            st.code(raw_tokens, language="python")
            
        with col2:
            st.markdown("#### 🟢 After (Normalized Tokens)")
            st.code(smart_tokens, language="python")
            
        # กล่องอธิบายการทำงาน (กดลูกศรเพื่อเปิด-ปิดได้ หน้าเว็บจะได้ไม่รก)
        with st.expander("ℹ️ How it works under the hood"):
            st.write("1. **Regex Reduction:** Identifies and reduces characters repeated 3 or more times.")
            st.write("2. **Tokenization:** Splits the text into words using PyThaiNLP's `newmm` engine.")
            st.write("3. **Spell Correction:** Maps invalid words to the closest valid Thai word using dictionary-based correction algorithms.")
            
    else:
        st.warning("⚠️ Please enter some text before processing.")