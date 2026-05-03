"""
Seq2Seq inference for Thai text normalization.
Vocabularies are cached to disk after first build to avoid re-downloading the dataset.
"""

import pickle
import sys
import os
from pathlib import Path

import torch
from pythainlp.tokenize import word_tokenize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Vocabulary, Encoder, Decoder, Seq2Seq, device
from config import MODEL_PATH, VOCAB_CACHE_PATH

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256


def load_or_build_vocabularies() -> tuple:
    """
    Load vocabs from cache file if it exists; otherwise download the HuggingFace
    dataset, build vocabs, and save to cache. Cache path: config.VOCAB_CACHE_PATH.
    """
    cache = Path(VOCAB_CACHE_PATH)
    if cache.exists():
        print("Loading vocabularies from cache...")
        with cache.open('rb') as f:
            return pickle.load(f)

    print("Building vocabularies from Hugging Face (first run only)...")
    from datasets import load_dataset
    dataset = load_dataset("thassanawalai/thai-slang-parallel-corpus", split="train")

    source_vocab = Vocabulary("noisy_thai")
    target_vocab = Vocabulary("formal_thai")
    for src, trg in zip(dataset['noisy_text'], dataset['formal_text']):
        source_vocab.add_sentence(src)
        target_vocab.add_sentence(trg)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('wb') as f:
        pickle.dump((source_vocab, target_vocab), f)
    print(f"Vocabularies cached at {cache}")

    return source_vocab, target_vocab


def load_trained_model(source_vocab_size: int, target_vocab_size: int) -> Seq2Seq:
    """Instantiate and load weights from config.MODEL_PATH."""
    enc = Encoder(source_vocab_size, HID_DIM, ENC_EMB_DIM).to(device)
    dec = Decoder(target_vocab_size, HID_DIM, DEC_EMB_DIM).to(device)
    model = Seq2Seq(enc, dec, target_vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def normalize_text(model: Seq2Seq, sentence: str, source_vocab: Vocabulary,
                   target_vocab: Vocabulary, max_len: int = 50) -> str:
    """Run greedy decoding on a single Thai sentence."""
    model.eval()
    tokens = word_tokenize(sentence, engine='newmm')
    token_indices = [source_vocab.word2index.get(w, 3) for w in tokens] + [2]
    source_tensor = torch.tensor(token_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden = model.encoder(source_tensor)

    decoder_input = torch.tensor([1]).to(device)
    decoded_words = []

    for _ in range(max_len):
        output, hidden = model.decoder(decoder_input, hidden)
        predicted_idx = output.argmax(1).item()
        if predicted_idx == 2:  # <EOS>
            break
        decoded_words.append(target_vocab.index2word.get(predicted_idx, "<UNK>"))
        decoder_input = output.argmax(1)

    return "".join(decoded_words)


if __name__ == "__main__":
    source_vocab, target_vocab = load_or_build_vocabularies()
    model = load_trained_model(source_vocab.n_words, target_vocab.n_words)

    print("\nModel ready. Type 'exit' to quit.")
    print("-" * 50)
    while True:
        user_input = input("Enter noisy Thai text: ")
        if user_input.lower() == 'exit':
            break
        if user_input.strip():
            result = normalize_text(model, user_input, source_vocab, target_vocab)
            print(f"Normalized: {result}\n")
