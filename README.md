# Thai Text Normalizer System (ระบบประมวลผลและปรับแต่งข้อความภาษาไทย)

An automatic Thai text normalization system for Natural Language Processing (NLP). It is designed to handle informal social media text, slang, word elongation, and misspellings, converting them into proper formal sentences with an integrated audio pronunciation feature.

## Key Features (คุณสมบัติหลัก)

1. **Cloud-based Slang Dictionary:** * Fetches a custom Thai slang dataset directly from Hugging Face (`thassanawalai/thai-social-slang-dict`) using the `datasets` library.
2. **Rule-based and Dictionary Normalization:** * Reduces character elongation using regular expressions.
   * Performs custom tokenization and dictionary-based spelling correction using `PyThaiNLP` and standard Thai lexicons, ensuring valid words remain unaltered.
3. **Text-to-Speech Integration:** * Automatically generates audio pronunciation of the normalized text using `gTTS` (Google Text-to-Speech).
4. **Interactive User Interface:** * Built with `Streamlit`.
   * Professional dark-themed UI featuring a clear comparison of raw input versus normalized tokens and a built-in audio player.

## System Architecture (โครงสร้างของระบบ)

* `app.py`: The main Streamlit application containing the user interface, normalization logic, and text-to-speech generation.
* `requirements.txt`: List of required Python dependencies for deployment.

## Installation and Usage (การติดตั้งและการใช้งาน)

**1. Install required libraries (ติดตั้งไลบรารีที่จำเป็น):**

```bash
pip install streamlit pandas pythainlp datasets gTTS