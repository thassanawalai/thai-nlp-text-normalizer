import streamlit as st
import re
from pythainlp.tokenize import word_tokenize

# --- ฟังก์ชันพระเอกของเรา (ลอจิกเดิมเป๊ะๆ) ---
def normalize_elongated(text):
    normalized_text = re.sub(r'(.)\1{2,}', r'\1', text)
    return normalized_text

# --- ส่วนออกแบบหน้าเว็บ ---
st.title("✨ Thai NLP: Word Elongation Handler")
st.markdown("บอกลาปัญหาพิมพ์ลากเสียงยาว เช่น **'สวัสดีค้าบบบบ'** ให้ AI ตัดคำได้แม่นยำขึ้น!")

# ช่องรับข้อความจากผู้ใช้
user_input = st.text_input("💬 พิมพ์ข้อความที่ต้องการทดสอบตรงนี้เลย:", "สวัสดีค้าบบบบ วันนี้อากาศดีจังเลยยยยย")

# ปุ่มกดประมวลผล
if st.button("🚀 รันเลย!"):
    if user_input:
        st.divider() # เส้นคั่น
        
        # 1. โชว์ผลลัพธ์แบบยังไม่แก้
        st.subheader("🔴 ก่อนแก้ (Before Normalization):")
        before_tokens = word_tokenize(user_input, engine='newmm')
        st.write(before_tokens)

        # 2. คลีนข้อความ
        cleaned_text = normalize_elongated(user_input)
        st.subheader("✨ ข้อความที่คลีนแล้ว:")
        st.info(cleaned_text) # โชว์ในกล่องสีฟ้าสวยๆ

        # 3. โชว์ผลลัพธ์หลังแก้
        st.subheader("🟢 หลังแก้ (After Normalization):")
        after_tokens = word_tokenize(cleaned_text, engine='newmm')
        st.success(after_tokens) # โชว์ในกล่องสีเขียว
    else:
        st.warning("หนูต้องพิมพ์ข้อความก่อนน้าา!")