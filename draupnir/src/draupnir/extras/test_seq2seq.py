import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        # Linear layers for query, key, value
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        # hidden_states: [seq_len, batch_size, hidden_size]
        batch_size = hidden_states.size(1)

        # Compute query, key, value
        Q = self.query(hidden_states)  # [seq_len, batch_size, hidden_size]
        K = self.key(hidden_states)    # [seq_len, batch_size, hidden_size]
        V = self.value(hidden_states)  # [seq_len, batch_size, hidden_size]

        # Compute attention scores
        scores = torch.bmm(Q.transpose(0, 1), K.transpose(1, 0).transpose(2,1)) / (self.hidden_size ** 0.5)
        # scores: [batch_size, seq_len, seq_len]


        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.bmm(attn_weights, V.transpose(0, 1))  # [batch_size, seq_len, hidden_size]
        context = context.transpose(0, 1)  # [seq_len, batch_size, hidden_size]

        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        #self.fc_in = nn.Linear(input_dim,emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src.to(torch.int64))
        outputs, last_hidden = self.rnn(embedded)
        return outputs,last_hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden


class AttentionalRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(AttentionalRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        #self.fc_in = nn.Linear(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=False) #expacts [L, batch; feat_dim]
        self.attention = SelfAttention(hidden_size)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)  # *2 for context + hidden

    def forward(self, input, hidden, hidden_states):
        # input: [1, batch_size]
        # hidden: (h, c) for LSTM
        # hidden_states: [seq_len, batch_size, hidden_size]

        embedded = self.embedding(input)  # [1, batch_size, embed_size]
        if embedded.ndim < 3:
            embedded = embedded.unsqueeze(0)

        output, hidden = self.rnn(embedded, hidden)  # output: [1, batch_size, hidden_size]


        # Apply self-attention
        context, _ = self.attention(hidden_states)  # context: [seq_len, batch_size, hidden_size]
        # Use the last context vector (for the current step)
        context = context[-1:, :, :]  # [1, batch_size, hidden_size]

        # Concatenate output and context
        output = torch.cat((output, context), dim=-1)  # [1, batch_size, hidden_size * 2]
        output = self.fc_out(output)  # [1, batch_size, vocab_size]

        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=10, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        #trg_vocab_size = self.decoder.fc.out_features
        outputs = []

        hidden_states,last_hidden = self.encoder(src) #[1,batch_size,hidden_dim]

        #hidden_states = hidden_states.permute(1,0,2)

        input = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        for t in range(max_len):
            output, hidden = self.decoder(input, last_hidden,hidden_states) #why give zeroes?
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(0))

            if trg is not None and t < trg.shape[0] and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[t]
            else:
                input = top1

        outputs = torch.cat(outputs, dim=0)
        return outputs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

torch.set_default_dtype(torch.float64) #might not be necessary

VOCAB_SIZE = 10
EMB_DIM = 8
HID_DIM = 16
SEQ_LEN = 5
BATCH_SIZE = 20

enc = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM)
dec = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM)

dec = AttentionalRNNDecoder(VOCAB_SIZE, EMB_DIM, HID_DIM, num_layers=1)

model = Seq2Seq(enc, dec, device).to(device)

src = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)
trg = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)

outputs = model(src, trg, max_len=SEQ_LEN, teacher_forcing_ratio=0.7)

print("Source sequence (input tokens):")
print(src.T)
print("\nTarget sequence (true tokens):")
print(trg.T)
print("\nPredicted sequence (model output tokens):")
print(outputs.T)