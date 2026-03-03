import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from tqdm import tqdm # progress bar utility

# Enable tqdm integration for pandas
tqdm.pandas()

print("🚀 Preparing to build a parallel dataset (Parallel Corpus)...")

df = pd.read_csv('wisesight_raw.csv')

# Take a small sample (100 rows) for a quick test run
df_sample = df.head(100).copy()

def auto_clean_text(text):
    text = str(text)
    reduced = re.sub(r'(.)\1{2,}', r'\1', text)
    tokens = word_tokenize(reduced, engine='newmm')
    
    smart_tokens = []
    for word in tokens:
        # Safety guard: if a token is longer than 15 chars or contains
        # non-Thai characters, skip spell-correction to avoid noise
        if len(word) > 15 or not re.match(r'^[ก-๙]+$', word):
            smart_tokens.append(word)
        else:
            smart_tokens.append(correct(word))

    return "".join(smart_tokens)

print("🤖 Processing text to produce formalized versions... (see progress bar below)")

# Use progress_apply instead of apply to display a nice progress bar
df_sample['clean_text'] = df_sample['texts'].progress_apply(auto_clean_text)

df_sample = df_sample.rename(columns={'texts': 'noisy_text'})
df_sample[['noisy_text', 'clean_text']].to_csv('slang_dataset.csv', index=False, encoding='utf-8-sig')

print("-" * 50)
print("✅ Created slang_dataset.csv successfully!")
print(df_sample[['noisy_text', 'clean_text']].head(5))