import re
from pythainlp.tokenize import word_tokenize

def normalize_elongated(text):
    # กฎ Regex ยุบคำที่พิมพ์ซ้ำกันยาวๆ ให้เหลือตัวเดียว
    normalized_text = re.sub(r'(.)\1{2,}', r'\1', text)
    return normalized_text

# ประโยคปัญหาที่หนูเจอ
original_text = "สวัสดีค้าบบบบ วันนี้อากาศดีจังเลยยยยย หิวข้าววววววววววว"

print("🔴 ก่อนแก้ (Before Normalization):")
print(word_tokenize(original_text, engine='newmm'))
print("-" * 50)

# ทำการคลีนข้อความ
cleaned_text = normalize_elongated(original_text)
print("✨ ข้อความที่คลีนแล้ว:", cleaned_text)
print("-" * 50)

print("🟢 หลังแก้ (After Normalization):")
print(word_tokenize(cleaned_text, engine='newmm'))