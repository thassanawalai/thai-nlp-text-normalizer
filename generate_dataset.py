import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from tqdm import tqdm # 👈 อิมพอร์ตเครื่องมือสร้างหลอดโหลด

# เปิดใช้งานโหมดหลอดโหลดให้ Pandas
tqdm.pandas()

print("🚀 กำลังเตรียมสร้าง Dataset แบบคู่ขนาน (Parallel Corpus)...")

df = pd.read_csv('wisesight_raw.csv')

# เค้าขอลดจำนวนลงมาเหลือ 100 ประโยคก่อนน้า จะได้เทสต์ว่ารันผ่านไวๆ
df_sample = df.head(100).copy()

def auto_clean_text(text):
    text = str(text)
    reduced = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = word_tokenize(reduced, engine='newmm')
    
    smart_tokens = []
    for word in tokens:
        # 🛡️ ระบบป้องกัน AI ช็อก: 
        # ถ้าคำยาวเกิน 15 ตัวอักษร หรือ ไม่ใช่พยัญชนะไทยร้วนๆ ให้ข้ามการแก้คำผิดไปเลย
        if len(word) > 15 or not re.match(r'^[ก-๙]+$', word):
            smart_tokens.append(word)
        else:
            smart_tokens.append(correct(word))
            
    return "".join(smart_tokens)

print("🤖 กำลังให้ AI ตัวเก่าช่วยสร้างประโยคทางการ... (ดูหลอดโหลดด้านล่างเลย!)")

# ใช้ progress_apply แทน apply ธรรมดา เพื่อโชว์หลอดโหลดสวยๆ
df_sample['clean_text'] = df_sample['texts'].progress_apply(auto_clean_text)

df_sample = df_sample.rename(columns={'texts': 'noisy_text'})
df_sample[['noisy_text', 'clean_text']].to_csv('slang_dataset.csv', index=False, encoding='utf-8-sig')

print("-" * 50)
print("✅ สร้างไฟล์ slang_dataset.csv สำเร็จแล้ว รอดแล้วค่ะตัวเล็ก!")
print(df_sample[['noisy_text', 'clean_text']].head(5))