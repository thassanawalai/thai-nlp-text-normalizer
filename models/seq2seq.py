"""
Seq2Seq model architecture for Thai text normalization.
Shared module used by both training (scripts/train_seq2seq.py) and inference (models/inference.py).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pythainlp.tokenize import word_tokenize
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    def __init__(self, name: str):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4

    def add_sentence(self, sentence: str):
        for word in word_tokenize(sentence, engine='newmm'):
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, emb_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, emb_size: int, dropout_p: float = 0.1):
        super().__init__()
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
    def __init__(self, encoder: Encoder, decoder: Decoder, target_vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, teacher_forcing_ratio: float = 0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]

        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)
        hidden = self.encoder(source)
        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs


class ThaiParallelDataset(Dataset):
    """PyTorch dataset for noisy/formal Thai sentence pairs."""

    def __init__(self, source_sentences, target_sentences, source_vocab: Vocabulary, target_vocab: Vocabulary):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        src_tokens = word_tokenize(self.source_sentences[idx], engine='newmm')
        trg_tokens = word_tokenize(self.target_sentences[idx], engine='newmm')
        src_seq = [self.source_vocab.word2index.get(w, 3) for w in src_tokens] + [2]
        trg_seq = [self.target_vocab.word2index.get(w, 3) for w in trg_tokens] + [2]
        return torch.tensor(src_seq), torch.tensor(trg_seq)


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch
