import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct

# --- 1. การตั้งค่าหน้าเว็บ (ต้องอยู่บรรทัดแรกเสมอ) ---
st.set_page_config(
    page_title="Thai NLP Normalizer", 
    page_icon="⚙️", 
    layout="wide" # ขยายหน้าเว็บให้กว้างขึ้น ดูเป็นมืออาชีพ
)

# --- 2. ฟังก์ชันประมวลผลหลัก ---
def auto_normalize_text(text):
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = word_tokenize(reduced_text, engine='newmm')
    smart_tokens = [correct(word) for word in tokens]
    return "".join(smart_tokens)

# --- 3. แถบด้านข้าง (Sidebar) สำหรับข้อมูลโปรเจกต์ ---
with st.sidebar:
    st.header("เกี่ยวกับระบบ")
    st.markdown("""
    ระบบประมวลผลและปรับแต่งข้อความภาษาไทย (Thai Text Normalization) 
    พัฒนาขึ้นเพื่อแก้ไขปัญหา:
    - **Word Elongation:** การพิมพ์ตัวอักษรซ้ำ
    - **Misspelling:** การสะกดคำผิดหรือคำวิบัติ
    
    *กระบวนการทำงานอาศัยเทคนิค Regular Expression ควบคู่กับ Dictionary-based Spelling Correction จากคลังข้อมูลภาษาไทยแห่งชาติ (TNC)*
    """)
    st.divider()
    st.caption("Developed by: AI Undergraduate @ HCU")

# --- 4. ส่วนเนื้อหาหลัก (Main Content) ---
st.title("⚙️ Thai Text Normalizer System")
st.markdown("ระบบปรับแต่งและแก้ไขข้อความภาษาไทยอัตโนมัติสำหรับการประมวลผลทางภาษาธรรมชาติ (NLP)")
st.divider()

# ส่วนรับข้อมูล (Input)
st.subheader("ส่วนนำเข้าข้อมูล (Input)")
user_input = st.text_area("กรุณาระบุข้อความที่ต้องการประมวลผล:", height=100, placeholder="ระบุข้อความภาษาไทยที่นี่...")

# ปุ่มประมวลผล
if st.button("ประมวลผลข้อมูล (Process)", type="primary"):
    if user_input.strip():
        st.divider()
        st.subheader("ผลลัพธ์การประมวลผล (Output)")
        
        # ใช้ Column แบ่งซ้าย-ขวา เพื่อเปรียบเทียบ
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ก่อนการปรับแต่ง (Raw Input Tokens)**")
            raw_tokens = word_tokenize(user_input, engine='newmm')
            # ใช้ st.code เพื่อแสดงผลลัพธ์แบบบล็อกโค้ด (ดูเป็น Developer)
            st.code(raw_tokens, language="python")
            
        with col2:
            st.markdown("**หลังการปรับแต่ง (Normalized Tokens)**")
            cleaned_text = auto_normalize_text(user_input)
            smart_tokens = word_tokenize(cleaned_text, engine='newmm')
            st.code(smart_tokens, language="python")
            
        # แสดงประโยคที่สมบูรณ์
        st.info(f"ข้อความที่ปรับแต่งแล้ว: {cleaned_text}")
    else:
        st.error("กรุณาระบุข้อความก่อนทำการประมวลผล")