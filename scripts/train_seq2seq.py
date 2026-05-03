"""
Training script for the Thai text normalization Seq2Seq model.
Run from project root: python scripts/train_seq2seq.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from models.seq2seq import (
    Vocabulary, Encoder, Decoder, Seq2Seq,
    ThaiParallelDataset, collate_fn, device,
)
from config import MODEL_PATH

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32


def train():
    print(f"Device: {device}")

    print("Downloading dataset from Hugging Face...")
    try:
        dataset = load_dataset("thassanawalai/thai-slang-parallel-corpus", split="train")
        source_texts = dataset['noisy_text']
        target_texts = dataset['formal_text']
        print(f"Loaded {len(source_texts)} sentence pairs.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Building vocabularies...")
    source_vocab = Vocabulary("noisy_thai")
    target_vocab = Vocabulary("formal_thai")
    for src, trg in zip(source_texts, target_texts):
        source_vocab.add_sentence(src)
        target_vocab.add_sentence(trg)
    print(f"Source vocab: {source_vocab.n_words} | Target vocab: {target_vocab.n_words}")

    train_dataset = ThaiParallelDataset(source_texts, target_texts, source_vocab, target_vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    enc = Encoder(source_vocab.n_words, HID_DIM, ENC_EMB_DIM).to(device)
    dec = Decoder(target_vocab.n_words, HID_DIM, DEC_EMB_DIM).to(device)
    model = Seq2Seq(enc, dec, target_vocab.n_words).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Training...")
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1:02} | Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'.")


if __name__ == "__main__":
    train()
