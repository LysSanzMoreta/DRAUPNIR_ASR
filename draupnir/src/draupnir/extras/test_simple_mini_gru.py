import os
import random
from string import ascii_letters
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seq_len = 6
n_seqs = 1000
batch_size = 100
input_size = 3
hidden_size = 10
embedding_size = 5
vocab_size = 50 + seq_len + 1 #should be the max number of numbers that we have, so that is 50 + seq_len + 1

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """https://d2l.ai/chapter_recurrent-modern/gru.html"""
        super().__init__()
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h)) #update gate
        r = torch.sigmoid(self.Wr(x) + self.Ur(h)) #reset gate
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h))
        h = (1 - z) * h + z * h_tilde
        return h

class FastGRUCell(nn.Module): #faster fused approach
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.Wx = nn.Linear(input_size, 3 * hidden_size)
        self.Uh = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.Urh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        Wx = self.Wx(x)
        Uh = self.Uh(h)


        Wx_z, Wx_r, Wx_h = Wx.chunk(3, dim=-1)
        Uh_z, Uh_r = Uh.chunk(2, dim=-1) #last part ignored? it has been already multiplied by h

        z = torch.sigmoid(Wx_z + Uh_z)
        r = torch.sigmoid(Wx_r + Uh_r)

        # IMPORTANT: recompute only candidate part with reset
        h_tilde = torch.tanh(Wx_h + self.Urh(r * h))

        h = (1 - z) * h + z * h_tilde
        return h

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,max_len = None):
        super().__init__()
        #self.cell = GRUCell(input_size, hidden_size)
        self.cell = FastGRUCell(input_size, hidden_size)

    def forward(self, x, h=None): #this code is for batch_size in second dimension
        # x: (seq_len, batch, input_size)

        x = x.transpose(1,0)
        seq_len, batch_size, _ = x.shape

        if h is None:
            h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h = self.cell(x[t], h) #hidden state at this time step, combine current input and previous hidden state
            outputs.append(h)

        out = torch.stack(outputs)
        out = out.transpose(1,0)

        return out, h


class miniGRU(nn.Module):
    def __init__(self, input_size, hidden_size,max_len = None):

        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.Z = nn.Linear(max_len*input_size, hidden_size*self.max_len)
        self.H = nn.Linear(max_len*input_size,hidden_size*self.max_len)



    def forward(self, x): #this code is for batch_size in second dimension

        nseqs = x.shape[0]

        x = x.flatten(1,2) #flatten last 2 dims

        z = self.Z(x) #[nseqs, max_len*hidden_size]
        h_tilde = self.H(x) #[nseqs, max_len*hidden_size] #all hidden candidates for all positions
        h_tilde = h_tilde.reshape(nseqs,self.max_len,self.hidden_size)
        z = z.reshape(nseqs,self.max_len,self.hidden_size)

        h_prev = h_tilde[:,:-1] # we add a dummy start hidden state and drop the last one --> hopefully, the last hidden state gets updated

        h_init = torch.zeros_like(h_prev[:,0])[:,None]
        h_prev = torch.concat([h_init,h_prev],axis=1)

        h = (1 - z) * h_prev + z * h_tilde

        h_last = h[:,-1]

        return h, h_last


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = seq_len
        self.embed = nn.Embedding(vocab_size, self.embedding_size,max_norm=1)
        #self.gru = GRU(self.embedding_size, self.hidden_size)
        self.gru = miniGRU(self.embedding_size, self.hidden_size,self.max_len)
        #self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):

        x = self.embed(x)
        out, _ = self.gru(x)
        out = self.fc(out)

        return out

class ToyDatasets():
    def __init__(self, dataset_name,n_samples,seq_len):

        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.seq_len = seq_len

    def generate_dataset(self):
        X = []
        Y = []

        start_idxs = np.random.randint(0,50,(self.n_samples))
        end_idxs = start_idxs +self.seq_len + 1

        for start_idx, end_idx in zip(start_idxs,end_idxs):
            seq = torch.arange(start_idx,end_idx)
            X.append(seq[:-1])
            Y.append(seq[1:])

        return torch.stack(X), torch.stack(Y)

    def copy_dataset(self, vocab=10):
        X = torch.randint(0, vocab, (self.n_samples, self.seq_len))
        Y = X.clone()
        return X, Y

    def fibonacci_dataset(self,n_samples=1000, seq_len=8, vocab=20):
        X = torch.zeros(self, n_samples, seq_len, dtype=torch.long)
        Y = torch.zeros(self,n_samples, seq_len, dtype=torch.long)

        for i in range(n_samples):
            a = torch.randint(0, vocab, (1,))
            b = torch.randint(0, vocab, (1,))
            seq = [a.item(), b.item()]

            for _ in range(seq_len):
                seq.append((seq[-1] + seq[-2]) % vocab)

            seq = torch.tensor(seq)

            X[i] = seq[:-1]
            Y[i] = seq[1:]

        return X, Y

    def build(self):
        if self.dataset_name == "sequential":
            return self.generate_dataset()



np.random.seed(0)
torch.manual_seed(0)
x, y = ToyDatasets("sequential",n_seqs,seq_len).build()
dataset = torch.utils.data.TensorDataset(x,y) #unsqueeze(-1) if there is no embeddings



#x = torch.randn(seq_len, batch, input_size)

model = GRUModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)



train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)


nepochs = 500

for epoch in range(nepochs):
    total_acc = 0
    total_loss = 0
    for x, y in train_loader:
        logits = model(x)
        loss = criterion(
            logits.view(-1, vocab_size), #[Nseqs, Ncategories] --> (batch_size*seq_len,vocab_size)
            y.view(-1) #[N] --> (batch*seq_len)
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(-1)
        acc = (preds == y).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

        print(
            f"epoch {epoch} "
            f"loss {total_loss/len(train_loader):.3f} "
            f"acc {total_acc/len(train_loader):.3f}"
        )


model.eval()

test_seq = torch.tensor([[20,21,22,23,24,25]])  # input,for the mini GRU, the train and test sequences need to have the same max_len
#test_seq = torch.tensor([[12,13,14,15,16,17]])  # input

assert test_seq.shape[1] == seq_len, "train and test must have same max seq len"

print("test seq",test_seq.shape)

with torch.no_grad():
    logits = model(test_seq)
    preds = logits.argmax(dim=-1)

print("Input: ", test_seq) #for each token in the input, we predict the next one [20->21, 21-> 22, ...]
print("Predicted:", preds) #should be [21,22,23,24,25]



