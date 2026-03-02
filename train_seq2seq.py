import pandas as pd
import torch
from pythainlp.tokenize import word_tokenize

# --- 1. คลาสสำหรับสร้างคลังคำศัพท์ (Vocab) ---
class Vocab:
    def __init__(self, name):
        self.name = name
        # กำหนด Token พิเศษที่ AI ต้องใช้
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # เริ่มต้นมี 4 คำ

    def add_sentence(self, sentence):
        # ตัดคำภาษาไทยแล้วเอาไปเก็บในคลัง
        for word in word_tokenize(str(sentence), engine='newmm'):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# --- 2. โหลดข้อมูลจาก Dataset ของเรา ---
print("กำลังโหลด Dataset...")
df = pd.read_csv('slang_dataset.csv')

# สร้างคลังคำศัพท์ 2 ฝั่ง (ฝั่งสแลง และ ฝั่งทางการ)
slang_vocab = Vocab("Slang")
formal_vocab = Vocab("Formal")

# วนลูปอ่านข้อมูลทุกบรรทัดเพื่อสอนให้คลังคำศัพท์รู้จักคำใหม่ๆ
for index, row in df.iterrows():
    slang_vocab.add_sentence(row['noisy_text'])
    formal_vocab.add_sentence(row['clean_text'])

print(f"✅ ฝั่งคำสแลง (Input) มีคำศัพท์ทั้งหมด: {slang_vocab.n_words} คำ")
print(f"✅ ฝั่งคำทางการ (Output) มีคำศัพท์ทั้งหมด: {formal_vocab.n_words} คำ")
print("-" * 30)
print(f"ตัวอย่างคำศัพท์สแลง: {list(slang_vocab.word2index.keys())[4:10]}")

import torch.nn as nn
import torch.nn.functional as F

# กำหนดขนาดความจำของสมอง (ยิ่งเยอะยิ่งฉลาด แต่เทรนนาน)
hidden_size = 256

# ==========================================
# 🧠 3. สร้างสมองส่วนอ่าน (Encoder)
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding: แปลง "ตัวเลข index" ให้เป็น "เวกเตอร์ความหมาย" (ให้ AI เข้าใจความหมายคำ)
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # GRU: สมองส่วนที่ทำหน้าที่จำลำดับคำในประโยค
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor, hidden_tensor):
        # 1. แปลงคำเป็นเวกเตอร์
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        # 2. ส่งให้ GRU อ่านและจำ
        output, hidden = self.gru(embedded, hidden_tensor)
        return output, hidden

    def initHidden(self):
        # เคลียร์ความจำก่อนเริ่มอ่านประโยคใหม่
        return torch.zeros(1, 1, self.hidden_size)

# ==========================================
# ✍️ 4. สร้างสมองส่วนเขียน (Decoder)
# ==========================================
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Linear: ชั้นสุดท้ายสำหรับแปลงเวกเตอร์ความหมาย กลับเป็น "คำศัพท์ภาษาไทย"
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = F.relu(embedded) # กระตุ้นเซลล์สมองด้วย ReLU
        
        output, hidden = self.gru(embedded, hidden_tensor)
        
        # คำนวณความน่าจะเป็นว่าคำต่อไปควรจะเป็นคำว่าอะไร
        output = self.softmax(self.out(output[0]))
        return output, hidden

# --- ทดลองสร้างตัวโมเดลขึ้นมาจริงๆ ---
print("กำลังสร้างสถาปัตยกรรมโมเดล...")
encoder = EncoderRNN(input_size=slang_vocab.n_words, hidden_size=hidden_size)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=formal_vocab.n_words)

print("✅ สร้าง Encoder และ Decoder สำเร็จแล้ว!")
print(encoder)

import torch.optim as optim

# ==========================================
# 🛠️ 5. ฟังก์ชันแปลงประโยคเป็นชุดตัวเลข (Tensors)
# ==========================================
def indexesFromSentence(vocab, sentence):
    # ตัดคำแล้วแปลงเป็นตัวเลขตามดิกชันนารีที่เราสร้างไว้
    return [vocab.word2index[word] for word in word_tokenize(str(sentence), engine='newmm') if word in vocab.word2index]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(vocab.word2index["<EOS>"]) # แปะป้าย <EOS> เพื่อบอกว่าจบประโยคแล้ว
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# ==========================================
# 🏋️ 6. ฟังก์ชันฝึกฝน AI 1 รอบ (Forward & Backward)
# ==========================================
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    # ล้างค่าความจำเดิมก่อนเริ่มเรียนรู้ใหม่
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    # 1. ฝั่ง Encoder: อ่านประโยควัยรุ่นทีละคำ
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # 2. ฝั่ง Decoder: เตรียมพ่นประโยคทางการ
    decoder_input = torch.tensor([[formal_vocab.word2index["<SOS>"]]]) # เริ่มด้วย <SOS>
    decoder_hidden = encoder_hidden # โอนความจำจาก Encoder มาให้ Decoder

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # เลือกคำที่ AI มั่นใจที่สุด
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        # คำนวณว่าเดาผิดไปแค่ไหน
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == formal_vocab.word2index["<EOS>"]:
            break

    # 3. Backpropagation: เรียนรู้จากความผิดพลาดและอัปเดตสมอง!
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# ==========================================
# 🚀 7. ลูปการฝึกฝนหลัก (Training Loop)
# ==========================================
print("เริ่มกระบวนการฝึกฝน AI (Training)...")
learning_rate = 0.01 # อัตราการเรียนรู้ (ห้ามเยอะไปเดี๋ยว AI งง)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss() # เครื่องมือวัดความผิดพลาด

epochs = 100 # ให้ AI ทำแบบฝึกหัดวนไป 100 รอบ

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
    
    # รายงานผลทุกๆ 10 รอบ
    if epoch % 10 == 0:
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch}/{epochs} | Loss (ความผิดพลาด): {avg_loss:.4f}")

print("✅ ฝึกฝนเสร็จสิ้น! AI ของเราฉลาดขึ้นแล้วค่ะ")

# ==========================================
# 🔮 8. ฟังก์ชันทดสอบความฉลาดของ AI (Evaluation)
# ==========================================
def evaluate(encoder, decoder, sentence, max_length=20):
    with torch.no_grad(): # ปิดระบบเรียนรู้ (เพราะตอนนี้เราแค่จะทดสอบมัน)
        # 1. แปลงประโยคที่ต้องการทดสอบให้เป็นตัวเลข
        input_tensor = tensorFromSentence(slang_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        # 2. ให้ Encoder อ่านและทำความเข้าใจ
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        # 3. เตรียมให้ Decoder เขียนประโยคตอบกลับ
        decoder_input = torch.tensor([[formal_vocab.word2index["<SOS>"]]])
        decoder_hidden = encoder_hidden # ส่งความจำจาก Encoder มาให้
        decoded_words = [] # ตะกร้าเก็บคำศัพท์ที่แปลแล้ว

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # เลือกคำที่ได้คะแนนสูงสุด
            topv, topi = decoder_output.data.topk(1)
            
            # ถ้าเจอคำว่า <EOS> แปลว่าจบประโยคแล้ว ให้หยุดพ่นคำ
            if topi.item() == formal_vocab.word2index["<EOS>"]:
                break 
            else:
                # แปลงตัวเลขกลับเป็นคำศัพท์ภาษาไทย
                decoded_words.append(formal_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # ประกอบคำกลับเป็นประโยคยาวๆ
        return "".join(decoded_words)

# ==========================================
# ✨ ลองของจริง! โยนคำถามท้าทาย AI
# ==========================================
print("\n" + "="*40)
print("✨ ทดสอบผลงาน: AI Text Normalizer ✨")
print("="*40)

# ลองเอาประโยคใน Dataset มาทดสอบดูว่ามันจำได้ไหม!
test_sentences = [
    "อ้วงจ๋าเค้าหิวข้าวววว",
    "ทำรายอยู่หยอออ",
    "คิดถึงจุงงงง"
]

for sentence in test_sentences:
    output = evaluate(encoder, decoder, sentence)
    print(f"🔴 ประโยควัยรุ่น : {sentence}")
    print(f"🟢 AI แปลทางการ : {output}")
    print("-" * 40)