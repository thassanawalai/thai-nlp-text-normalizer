from datasets import load_dataset
import pandas as pd

print("กำลังเชื่อมต่อและดาวน์โหลดข้อมูลจาก Hugging Face... รอแป๊บนึงน้า 🚀")

# 1. โหลดชุดข้อมูล Wisesight Sentiment จาก Hugging Face
dataset = load_dataset("wisesight_sentiment")

# 2. ดึงข้อมูลส่วนที่เป็น Training Set มาทำเป็นตาราง (DataFrame) ให้ดูง่ายๆ
df = pd.DataFrame(dataset['train'])

# 3. โชว์ผลลัพธ์ข้อความดิบๆ จากโซเชียล 10 บรรทัดแรก
print("\n✨ ตัวอย่างข้อความโซเชียลมีเดียของคนไทย (Noisy Text) ✨")
print("-" * 50)
for i, text in enumerate(df['texts'].head(10)):
    print(f"[{i+1}] {text}")
print("-" * 50)

# 4. (แถมให้!) เซฟข้อมูลออกมาเป็นไฟล์ CSV เผื่อหนูอยากเอาไปเปิดดูแบบเต็มๆ
df[['texts']].to_csv('wisesight_raw.csv', index=False, encoding='utf-8-sig')
print("✅ เซฟข้อมูลลงไฟล์ wisesight_raw.csv เรียบร้อยแล้วค่ะตัวเล็ก!")