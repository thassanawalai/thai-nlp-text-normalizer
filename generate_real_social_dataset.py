import pandas as pd
import time
from datasets import load_dataset
from google import genai
from tqdm import tqdm

# ==========================================
# 1. API CONFIGURATION
# ==========================================
# 🚨 คำเตือน: ตอนเอาขึ้น Git อย่าลืมลบ API Key ออกก่อนนะครับ!
GEMINI_API_KEY = "ใส่_API_KEY_ของคุณที่นี่"
client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. PROMPT TEMPLATE FOR LLM TRANSLATION
# ==========================================
def normalize_text_with_llm(social_text):
    prompt = f"""
    You are a linguistic expert in the Thai language. 
    Convert the following informal Thai social media text (which may contain slang, typos, word elongation, or missing vowels) into a formal, grammatically correct Thai sentence.
    Do not change the original core meaning. 
    Do not add any explanations, conversational fillers, or quotation marks. Output ONLY the formal Thai sentence.
    
    Informal Text: {social_text}
    Formal Text:
    """
    
    # ระบบ Smart Retry: พยายามซ้ำสูงสุด 5 ครั้งถ้าโดนบล็อกโควตา
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            # ถ้าเป็น Error โควตา (429) ให้หยุดพัก 60 วินาที
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"\n⏳ [ติดลิมิต] ระบบพัก 60 วินาทีเพื่อรีเซ็ตโควตา... (ลองใหม่ครั้งที่ {attempt + 1}/{max_retries})")
                time.sleep(60.0)
            else:
                print(f"\n❌ API Error: {error_str}")
                return None
                
    return None

# ==========================================
# 3. DATA EXTRACTION AND PROCESSING
# ==========================================
def generate_parallel_corpus(num_samples=1000):
    print("Loading 'wisesight_sentiment' dataset from Hugging Face...")
    try:
        dataset = load_dataset("wisesight_sentiment", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame()
    
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    parallel_data = []
    
    print(f"Starting LLM translation for {num_samples} samples. This may take a while...")
    for row in tqdm(dataset):
        noisy_text = row['texts'].strip()
        
        if len(noisy_text) < 5:
            continue
            
        formal_text = normalize_text_with_llm(noisy_text)
        
        if formal_text:
            parallel_data.append({
                "noisy_text": noisy_text,
                "formal_text": formal_text
            })
            
        # พัก 4 วินาที เพื่อให้ได้อัตราส่วน ~15 requests/minute แบบปลอดภัย
        time.sleep(4.0) 
        
    return pd.DataFrame(parallel_data)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TARGET_SAMPLES = 2000 
    
    df = generate_parallel_corpus(num_samples=TARGET_SAMPLES)
    
    if not df.empty:
        output_filename = "real_social_slang_dataset.csv"
        df.to_csv(output_filename, index=False, encoding="utf-8")
        
        print(f"\n✅ Successfully generated {len(df)} parallel sentences.")
        print(f"Saved dataset to '{output_filename}'.")
    else:
        print("\nFailed to generate the dataset.")