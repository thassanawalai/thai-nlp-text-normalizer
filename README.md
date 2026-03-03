# ⚙️ Thai Text Normalizer System

An automatic Thai text normalization system for Natural Language Processing (NLP). It is designed to handle informal social media text such as repeated characters (word elongation), slang, and misspellings.
## 📌 Key Features

1. **Rule-based & Dictionary Normalization:** Uses regular expressions to handle word elongation.
   * Performs dictionary-based spelling correction using `PyThaiNLP` and national Thai lexicons where available.
2. **Deep Learning Translation (Seq2Seq Model):** Sequence-to-sequence (Encoder-Decoder) model implemented with `PyTorch`.
   * Trained to translate informal/slang fragments into well-formed formal sentences.
3. **Interactive User Interface:** Built with `Streamlit`.
   * Professional/dark-themed UI for clear comparison of raw input vs normalized tokens.

## 🛠️ System Structure

* `app.py`: Main Streamlit UI and rule-based normalizer
* `train_seq2seq.py`: Seq2Seq model architecture and training loop (PyTorch)
* `generate_dataset.py`: Script to generate a synthetic parallel corpus (slang -> formal)
* `slang_dataset.csv`: Training dataset (parallel pairs)
## 🚀 Installation & Usage

**1. Install required libraries:**
```bash
pip install torch pandas pythainlp streamlit tqdm datasets
```
# ⚙️ Thai Text Normalizer System

ระบบประมวลผลและปรับแต่งข้อความภาษาไทยอัตโนมัติ (Thai Text Normalization) สำหรับงานด้านการประมวลผลภาษาธรรมชาติ (Natural Language Processing - NLP) พัฒนาขึ้นเพื่อจัดการกับข้อความบนโซเชียลมีเดียที่มีความไม่เป็นทางการ เช่น การพิมพ์ตัวอักษรซ้ำ (Word Elongation) และการใช้คำสแลงหรือคำวิบัติ (Misspelling/Slang)

## 📌 คุณสมบัติหลัก (Key Features)

1. **Rule-based & Dictionary Normalization:** * จัดการปัญหากาลากเสียง (Word Elongation) ด้วย Regular Expression
   * ตรวจสอบและแก้ไขคำสะกดผิดด้วย Dictionary-based Spelling Correction จากฐานข้อมูลคลังข้อมูลภาษาไทยแห่งชาติ (TNC) ผ่านไลบรารี `PyThaiNLP`
2. **Deep Learning Translation (Seq2Seq Model):** * โมเดลสถาปัตยกรรม Sequence-to-Sequence (Encoder-Decoder) พัฒนาด้วย `PyTorch` 
   * ฝึกสอน (Training) เพื่อทำหน้าที่แปลก้อนข้อความภาษาวัยรุ่นหรือคำสแลง ให้เป็นประโยคภาษาทางการที่มีความหมายสมบูรณ์
3. **Interactive User Interface:** * หน้าจอแสดงผลแบบโต้ตอบ พัฒนาด้วย `Streamlit` 
   * ออกแบบ UI สไตล์ Professional/Dark Mode เพื่อเปรียบเทียบข้อความก่อนและหลังการประมวลผล (Raw Input vs Normalized Tokens) ได้อย่างชัดเจน

## 🛠️ โครงสร้างของระบบ (System Architecture)

* `app.py`: ไฟล์หลักสำหรับรันหน้าเว็บแอปพลิเคชัน (Streamlit UI) และระบบ Rule-based Normalizer
* `train_seq2seq.py`: สถาปัตยกรรมโมเดล Seq2Seq และกระบวนการ Training Loop ด้วย PyTorch
* `generate_dataset.py`: สคริปต์สำหรับสร้าง Synthetic Parallel Corpus (Slang to Formal) จากชุดข้อมูลสาธารณะ
* `slang_dataset.csv`: ชุดข้อมูลสำหรับฝึกสอนโมเดล (Training Dataset)

## 🚀 การติดตั้งและการใช้งาน (Installation & Usage)

**1. ติดตั้งไลบรารีที่จำเป็น:**
```bash
pip install torch pandas pythainlp streamlit tqdm datasets