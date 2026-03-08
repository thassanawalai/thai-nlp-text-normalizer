import pandas as pd
import random
import os
from datasets import load_dataset

# ==========================================
# 1. LOAD DICTIONARY
# ==========================================
def load_slang_dictionary():
    # Load formal and slang words from your Hugging Face dataset
    dataset = load_dataset("thassanawalai/thai-social-slang-dict", split="train")
    return dict(zip(dataset['formal'], dataset['slang']))

# ==========================================
# 2. LOAD BASE SENTENCES FROM FILE
# ==========================================
def load_base_sentences(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        print("Please create the file and add formal Thai sentences line by line.")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as file:
        # Read lines and remove whitespace/newlines
        sentences = [line.strip() for line in file.readlines() if line.strip()]
    return sentences

# ==========================================
# 3. DATA AUGMENTATION PROCESS
# ==========================================
def generate_synthetic_data(base_sentences, num_samples=1000):
    slang_dict = load_slang_dictionary()
    formal_words = list(slang_dict.keys())
    
    synthetic_pairs = []
    
    while len(synthetic_pairs) < num_samples:
        sentence = random.choice(base_sentences)
        noisy_sentence = sentence
        
        # Inject slang words randomly
        for formal_word in formal_words:
            if formal_word in noisy_sentence and random.random() > 0.3:
                noisy_sentence = noisy_sentence.replace(formal_word, slang_dict[formal_word])
        
        # Inject character elongation (e.g., repeating vowels or consonants)
        if random.random() > 0.5 and len(noisy_sentence) > 5:
            idx = random.randint(0, len(noisy_sentence) - 1)
            char = noisy_sentence[idx]
            # Ensure we only repeat Thai characters
            if '\u0E00' <= char <= '\u0E7F': 
                noisy_sentence = noisy_sentence[:idx] + char * random.randint(2, 4) + noisy_sentence[idx+1:]
        
        # Only keep pairs that were successfully modified
        if noisy_sentence != sentence:
            synthetic_pairs.append({
                "noisy_text": noisy_sentence,
                "formal_text": sentence
            })
            
    return pd.DataFrame(synthetic_pairs)

if __name__ == "__main__":
    input_filename = "formal_sentences.txt"
    output_filename = "slang_dataset.csv"
    target_samples = 5000  # Adjust this number based on your needs
    
    print(f"Loading base sentences from '{input_filename}'...")
    base_sentences = load_base_sentences(input_filename)
    
    if not base_sentences:
        print("Process terminated due to missing input file.")
    else:
        print(f"Loaded {len(base_sentences)} base sentences.")
        print(f"Generating synthetic dataset with {target_samples} samples...")
        
        df = generate_synthetic_data(base_sentences, num_samples=target_samples)
        
        # Save the generated dataset to a CSV file
        df.to_csv(output_filename, index=False, encoding="utf-8")
        
        print(f"Dataset generated successfully.")
        print(f"Total samples: {len(df)}")
        print(f"Saved as '{output_filename}'.")