# ==========================================
# 🧠 3. CORE PROCESSING FUNCTION (อัปเกรดความฉลาด!)
# ==========================================
# สร้างพจนานุกรมสอน AI ให้รู้จักคำสแลงยอดฮิต
import pandas as pd

# ==========================================
# 📚 3. ระบบโหลดฐานข้อมูลคำสแลง (แยกไฟล์)
# ==========================================
@st.cache_data
def load_slang_dict():
    try:
        # ให้ Pandas ไปอ่านไฟล์ CSV แล้วแปลงเป็น Dictionary ให้เราอัตโนมัติ!
        df = pd.read_csv('slang_dict.csv', encoding='utf-8')
        # จับคู่คอลัมน์ slang กับ formal เข้าด้วยกัน
        return dict(zip(df['slang'], df['formal']))
    except FileNotFoundError:
        st.error("⚠️ หาไฟล์ slang_dict.csv ไม่เจอค่ะ อย่าลืมสร้างไฟล์น้า!")
        return {}

# โหลดดิกชันนารีมาเก็บไว้ในตัวแปร
slang_dict = load_slang_dict()

# ==========================================
# 🧠 4. CORE PROCESSING FUNCTION
# ==========================================
@st.cache_data
def auto_normalize_text(text):
    # 1. ยุบคำลากเสียง
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 2. ตัดคำ
    tokens = word_tokenize(reduced_text, engine='newmm')
    
    smart_tokens = []
    for word in tokens:
        # 3. ถ้าเป็นช่องว่าง (Space) หรือสัญลักษณ์ ให้ปล่อยผ่าน
        if not word.strip() or not re.match(r'^[ก-๙]+$', word):
            smart_tokens.append(word)
            
        # 4. 🌟 AI เช็คฐานข้อมูล: ถ้าเจอสแลงในไฟล์ CSV แปลงร่างทันที!
        elif word in slang_dict:
            smart_tokens.append(slang_dict[word])
            
        # 5. แก้คำผิดด้วย PyThaiNLP
        elif len(word) > 1:
            smart_tokens.append(correct(word))
            
        # 6. ป้องกันพยัญชนะเดี่ยว
        else:
            smart_tokens.append(word)
            
    return "".join(smart_tokens), tokens, smart_tokens

@st.cache_data
def auto_normalize_text(text):
    # 1. ยุบคำลากเสียง (เช่น ทำรายยยย -> ทำราย)
    reduced_text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 2. ตัดคำ
    tokens = word_tokenize(reduced_text, engine='newmm')
    
    smart_tokens = []
    for word in tokens:
        # 3. ถ้าเป็นช่องว่าง (Space) หรือสัญลักษณ์พิเศษ ให้ปล่อยผ่านไปเลย ห้ามแก้!
        if not word.strip() or not re.match(r'^[ก-๙]+$', word):
            smart_tokens.append(word)
            
        # 4. ถ้าเจอคำสแลงที่อยู่ใน Dictionary ของเรา ให้แปลงเป็นคำทางการทันที
        elif word in slang_dict:
            smart_tokens.append(slang_dict[word])
            
        # 5. ถ้าไม่ใช่สแลง และไม่ใช่คำเดี่ยวๆ ให้ PyThaiNLP ช่วยเช็คเผื่อสะกดผิด
        elif len(word) > 1:
            smart_tokens.append(correct(word))
            
        # 6. พยัญชนะโดดๆ (เช่น 'ส' ที่โดนตัดผิดมา) ให้ปล่อยผ่าน
        else:
            smart_tokens.append(word)
            
    return "".join(smart_tokens), tokens, smart_tokens