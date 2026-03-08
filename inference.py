import torch
import torch.nn as nn
from pythainlp.tokenize import word_tokenize
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. VOCABULARY AND ARCHITECTURE (Must match training exactly)
# ==========================================
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        tokens = word_tokenize(sentence, engine='newmm')
        for word in tokens:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

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
        
        x = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output
            
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs

# ==========================================
# 3. INITIALIZATION AND INFERENCE LOGIC
# ==========================================
def rebuild_vocabularies():
    print("Rebuilding vocabularies from Hugging Face for accurate mapping...")
    dataset = load_dataset("thassanawalai/thai-slang-parallel-corpus", split="train")
    source_texts = dataset['noisy_text']
    target_texts = dataset['formal_text']
    
    source_vocab = Vocabulary("noisy_thai")
    target_vocab = Vocabulary("formal_thai")
    
    for src, trg in zip(source_texts, target_texts):
        source_vocab.add_sentence(src)
        target_vocab.add_sentence(trg)
        
    return source_vocab, target_vocab

def load_trained_model(source_vocab_size, target_vocab_size):
    print("Loading model weights...")
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    
    enc = Encoder(source_vocab_size, HID_DIM, ENC_EMB_DIM).to(device)
    dec = Decoder(target_vocab_size, HID_DIM, DEC_EMB_DIM).to(device)
    model = Seq2Seq(enc, dec, target_vocab_size).to(device)
    
    # Load the state dictionary
    model.load_state_dict(torch.load('seq2seq_model.pt', map_location=device))
    model.eval() # Set model to evaluation mode
    return model

def normalize_text(model, sentence, source_vocab, target_vocab, max_len=50):
    model.eval()
    
    # Tokenize input
    tokens = word_tokenize(sentence, engine='newmm')
    
    # Convert words to indices
    token_indices = [source_vocab.word2index.get(word, 3) for word in tokens]
    token_indices.append(2) # Append <EOS>
    
    # Convert to tensor and add batch dimension
    source_tensor = torch.tensor(token_indices).unsqueeze(0).to(device)
    
    # Pass through encoder
    with torch.no_grad():
        hidden = model.encoder(source_tensor)
        
    # First input to decoder is <SOS>
    decoder_input = torch.tensor([1]).to(device)
    
    decoded_words = []
    
    for _ in range(max_len):
        output, hidden = model.decoder(decoder_input, hidden)
        
        # Get the highest predicted probability index
        top_prediction = output.argmax(1)
        predicted_idx = top_prediction.item()
        
        # Stop if <EOS> is generated
        if predicted_idx == 2:
            break
            
        # Add word to sequence
        decoded_word = target_vocab.index2word.get(predicted_idx, "<UNK>")
        decoded_words.append(decoded_word)
        
        # Feed prediction as next input
        decoder_input = top_prediction
        
    return "".join(decoded_words)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    import random # imported here for the dummy parameter in Seq2Seq if needed
    
    source_vocab, target_vocab = rebuild_vocabularies()
    model = load_trained_model(source_vocab.n_words, target_vocab.n_words)
    
    print("\nModel is ready! Type 'exit' to stop.")
    print("-" * 50)
    
    while True:
        user_input = input("Enter noisy Thai text: ")
        if user_input.lower() == 'exit':
            break
            
        if user_input.strip():
            result = normalize_text(model, user_input, source_vocab, target_vocab)
            print(f"Normalized Output: {result}\n")