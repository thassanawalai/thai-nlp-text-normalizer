import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
import random
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION AND DEVICE SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"System is configured to use device: {device}")

# ==========================================
# 2. VOCABULARY MANAGEMENT
# ==========================================
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        # Tokenize Thai sentences using PyThaiNLP
        tokens = word_tokenize(sentence, engine='newmm')
        for word in tokens:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# ==========================================
# 3. PYTORCH DATASET AND DATALOADER
# ==========================================
class ThaiParallelDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        src_tokens = word_tokenize(self.source_sentences[idx], engine='newmm')
        trg_tokens = word_tokenize(self.target_sentences[idx], engine='newmm')
        
        # Convert tokens to numerical indices, use <UNK> (3) if word is not in vocab
        src_seq = [self.source_vocab.word2index.get(w, 3) for w in src_tokens]
        trg_seq = [self.target_vocab.word2index.get(w, 3) for w in trg_tokens]
        
        # Append <EOS> token (2) at the end of sequences
        src_seq.append(2)
        trg_seq.append(2)
        
        return torch.tensor(src_seq), torch.tensor(trg_seq)

def collate_fn(batch):
    # Pad sequences to the maximum length in the batch
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# ==========================================
# 4. SEQ2SEQ MODEL ARCHITECTURE
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, emb_size, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        
        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)
        hidden = self.encoder(source)
        
        # First token is always <SOS> (1)
        x = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs

# ==========================================
# 5. TRAINING ROUTINE
# ==========================================
def train():
    # Load dataset from Hugging Face
    print("Downloading dataset from Hugging Face...")
    try:
        # User's specific parallel corpus repository
        dataset = load_dataset("thassanawalai/thai-slang-parallel-corpus", split="train")
        source_texts = dataset['noisy_text']
        target_texts = dataset['formal_text']
        print(f"Successfully loaded {len(source_texts)} sentence pairs.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Initialize and build vocabularies
    print("Building vocabularies...")
    source_vocab = Vocabulary("noisy_thai")
    target_vocab = Vocabulary("formal_thai")
    
    for src, trg in zip(source_texts, target_texts):
        source_vocab.add_sentence(src)
        target_vocab.add_sentence(trg)
        
    print(f"Source Vocabulary Size: {source_vocab.n_words}")
    print(f"Target Vocabulary Size: {target_vocab.n_words}")

    # Prepare DataLoader
    train_dataset = ThaiParallelDataset(source_texts, target_texts, source_vocab, target_vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Hyperparameters
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Instantiate models
    enc = Encoder(source_vocab.n_words, HID_DIM, ENC_EMB_DIM).to(device)
    dec = Decoder(target_vocab.n_words, HID_DIM, DEC_EMB_DIM).to(device)
    model = Seq2Seq(enc, dec, target_vocab.n_words).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Ignore <PAD> index (0) in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    print("Starting Model Training...")
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False)
        for src, trg in progress_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            output = model(src, trg)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1:02} | Average Training Loss: {avg_loss:.4f}")

    # Save the trained model weights
    save_path = "seq2seq_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to '{save_path}'.")

if __name__ == "__main__":
    train()
    