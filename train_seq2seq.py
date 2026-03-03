import pandas as pd
import torch
from pythainlp.tokenize import word_tokenize

# --- 1. Vocab class ---
class Vocab:
    def __init__(self, name):
        self.name = name
        # Special tokens required by the model
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # initial count (includes special tokens)

    def add_sentence(self, sentence):
        # Tokenize the sentence and add words to the vocab
        for word in word_tokenize(str(sentence), engine='newmm'):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# --- 2. Load dataset ---
print("Loading dataset...")
df = pd.read_csv('slang_dataset.csv')

# Create vocabularies for both sides (slang / formal)
slang_vocab = Vocab("Slang")
formal_vocab = Vocab("Formal")

# Populate vocabularies by iterating through the dataset
for index, row in df.iterrows():
    slang_vocab.add_sentence(row['noisy_text'])
    formal_vocab.add_sentence(row['clean_text'])

print(f"✅ Slang vocab (input) size: {slang_vocab.n_words}")
print(f"✅ Formal vocab (output) size: {formal_vocab.n_words}")
print("-" * 30)
print(f"Example slang tokens: {list(slang_vocab.word2index.keys())[4:10]}")

import torch.nn as nn
import torch.nn.functional as F

# Hidden size of the model (larger -> more capacity, slower training)
hidden_size = 256

# ==========================================
# 🧠 3. Encoder
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding: convert index to dense vector representation
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # GRU: recurrent unit to process sequence information
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor, hidden_tensor):
        # 1. แปลงคำเป็นเวกเตอร์
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        # 2. ส่งให้ GRU อ่านและจำ
        output, hidden = self.gru(embedded, hidden_tensor)
        return output, hidden

    def initHidden(self):
        # Initialize hidden state before processing a new sequence
        return torch.zeros(1, 1, self.hidden_size)

# ==========================================
# ✍️ 4. Decoder
# ==========================================
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Linear: final layer that maps to vocabulary logits
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = F.relu(embedded) # apply ReLU activation
        
        output, hidden = self.gru(embedded, hidden_tensor)
        
        # Compute token probabilities for the next word
        output = self.softmax(self.out(output[0]))
        return output, hidden

# --- ทดลองสร้างตัวโมเดลขึ้นมาจริงๆ ---
print("Building model architecture...")
encoder = EncoderRNN(input_size=slang_vocab.n_words, hidden_size=hidden_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=formal_vocab.n_words)

print("✅ Encoder and Decoder created successfully!")
print(encoder)

import torch.optim as optim

# ==========================================
# 🛠️ 5. Utilities: convert sentences to tensors
# ==========================================
def indexesFromSentence(vocab, sentence):
    # Tokenize and convert words to indices using our vocab
    return [vocab.word2index[word] for word in word_tokenize(str(sentence), engine='newmm') if word in vocab.word2index]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(vocab.word2index["<EOS>"]) # append <EOS> to mark end of sentence
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# ==========================================
# 🏋️ 6. Single training step (forward & backward)
# ==========================================
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    # Zero gradients before backprop
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    # 1. Encoder: read the input sequence token-by-token
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # 2. Decoder: generate the target sequence
    decoder_input = torch.tensor([[formal_vocab.word2index["<SOS>"]]]) # start with <SOS>
    decoder_hidden = encoder_hidden # transfer encoder state to decoder

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # Select the token with highest probability
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        # Compute loss for this step
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == formal_vocab.word2index["<EOS>"]:
            break

    # 3. Backpropagation: update model parameters
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# ==========================================
# 🚀 7. Training loop
# ==========================================
print("Starting training loop...")
learning_rate = 0.01 # learning rate
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss() # เครื่องมือวัดความผิดพลาด

epochs = 100 # number of training epochs

# แปลง Dataset ทั้งหมดให้เป็นตัวเลขรอไว้เลย
training_pairs = [(tensorFromSentence(slang_vocab, row['noisy_text']),
                   tensorFromSentence(formal_vocab, row['clean_text']))
                  for _, row in df.iterrows()]

# เริ่มเทรน!
for epoch in range(1, epochs + 1):
    total_loss = 0
    for input_tensor, target_tensor in training_pairs:
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss
    
    # Report progress every 10 epochs
    if epoch % 10 == 0:
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

print("✅ Training complete!")

# ==========================================
# 🔮 8. Evaluation
# ==========================================
def evaluate(encoder, decoder, sentence, max_length=20):
    with torch.no_grad():
        # 1. Convert input sentence to tensor
        input_tensor = tensorFromSentence(slang_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        # 2. Encoder processes the input sequence
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        # 3. Decoder generates the output sequence
        decoder_input = torch.tensor([[formal_vocab.word2index["<SOS>"]]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == formal_vocab.word2index["<EOS>"]:
                break
            else:
                # Convert index back to word
                decoded_words.append(formal_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return "".join(decoded_words)

# ==========================================
# ✨ Quick evaluation examples
# ==========================================
print("\n" + "="*40)
print("✨ Demo: AI Text Normalizer ✨")
print("="*40)

# Example test sentences (informal / slang)
test_sentences = [
    "im sooo hungry rn",
    "working on this lol",
    "miss u soooo much"
]

for sentence in test_sentences:
    output = evaluate(encoder, decoder, sentence)
    print(f"🔴 Input (slang): {sentence}")
    print(f"🟢 AI (formalized): {output}")
    print("-" * 40)