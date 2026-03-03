import streamlit as st
import re
from pythainlp.tokenize import word_tokenize

# --- Our main function (same logic) ---
def normalize_elongated(text):
    normalized_text = re.sub(r'(.)\1{2,}', r'\1', text)
    return normalized_text

# --- Web UI ---
st.title("✨ Thai NLP: Word Elongation Handler")
st.markdown("Fix elongated typing to improve AI tokenization accuracy!")

# User input field
user_input = st.text_input("💬 Enter text to test here:")

# Processing button
if st.button("🚀 Run Now!"):
    if user_input:
        st.divider() # Divider
        
        # 1. Show tokens before normalization
        st.subheader("🔴 Before Normalization:")
        before_tokens = word_tokenize(user_input, engine='newmm')
        st.write(before_tokens)

        # 2. Clean the text
        cleaned_text = normalize_elongated(user_input)
        st.subheader("✨ Cleaned Text:")
        st.info(cleaned_text) # Display in an info box

        # 3. Show tokens after normalization
        st.subheader("🟢 After Normalization:")
        after_tokens = word_tokenize(cleaned_text, engine='newmm')
        st.success(after_tokens) # Display in a success box
    else:
        st.warning("Please enter some text first!")