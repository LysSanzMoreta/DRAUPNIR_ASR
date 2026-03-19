import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GRUCell(nn.Module):
    def __init__(self, input_size: int |float, hidden_size: int |float):
        """
        Basic sequential implementation of the GRU cell

        https://d2l.ai/chapter_recurrent-modern/gru.html"""
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

class FastGRUCell(nn.Module):
    def __init__(self, input_size: int |float, hidden_size: int |float):
        """Faster fused approach of the GRU cell"""
        super().__init__()

        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, 3 * hidden_size)
        self.Uh = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.Urh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):

        Wx = self.Wx(x)
        Uh = self.Uh(h)

        Wx_z, Wx_r, Wx_h = Wx.chunk(3, dim=-1)
        Uh_z, Uh_r = Uh.chunk(2, dim=-1)

        z = torch.sigmoid(Wx_z + Uh_z)
        r = torch.sigmoid(Wx_r + Uh_r)

        # IMPORTANT: recompute only candidate part with reset
        h_tilde = torch.tanh(Wx_h + self.Urh(r * h))

        h = (1 - z) * h + z * h_tilde
        return h

class manualGRU(nn.Module):
    def __init__(self, input_size: int |float, hidden_size: int |float,max_len = int | None):
        super().__init__()
        #self.cell = GRUCell(input_size, hidden_size)
        self.cell = FastGRUCell(input_size, hidden_size)

    def forward(self, x, h=None):
        """Batch in second dimension approach"""
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
        self.input_size = input_size
        self.Z = nn.Linear(self.max_len*self.input_size, self.max_len*self.hidden_size)
        self.H = nn.Linear(self.max_len*self.input_size,self.max_len*self.hidden_size)

        self.num_layers = 1

    def forward(self, x, h_init=None): #this code is for batch_size in second dimension

        if x.shape[1] > self.max_len:
            excess_len = x.shape[1]-self.max_len
            x = x[:,excess_len:] #theoretically we need to see the entire sequence, but i am not sure how to do it in the minigru, so we chop
            #first_tokens = x[:,:excess_len]


        nseqs = x.shape[0]
        x = x.flatten(1,2) #flatten last 2 dims
        z = self.Z(x) #[nseqs, max_len*hidden_size]
        h_tilde = self.H(x) #[nseqs, max_len*hidden_size] #all hidden candidates for all positions
        h_tilde = h_tilde.reshape(nseqs,self.max_len,self.hidden_size)
        z = z.reshape(nseqs,self.max_len,self.hidden_size)

        h_prev = h_tilde[:,:-1] # we add a dummy start hidden state and drop the last one
        if h_init is not None:
            h_init = h_init.squeeze(0)[:, None]

        else:
            h_init = torch.zeros_like(h_prev[:,0])[:,None] #old basic version

        h_prev = torch.concat([h_init,h_prev],axis=1)
        h = (1 - z) * h_prev + z * h_tilde
        h_last = h[:,-1]

        return h, h_last


class GRUModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = seq_len
        self.embed = nn.Embedding(vocab_size, self.embedding_size,max_norm=1)
        self.bidirectional = True
        #self.gru = GRU(self.embedding_size, self.hidden_size)
        #self.gru = miniGRU(self.embedding_size, self.hidden_size,self.max_len)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_hidden =  nn.Parameter(torch.randn(self.hidden_size), requires_grad=True)
        self.num_layers = 1
        self.init_dims = 2 * self.num_layers if self.bidirectional else 1 * self.num_layers

    def forward(self, x):

        nseqs = x.shape[0]

        x = self.embed(x)
        h_init = self.init_hidden.expand(self.init_dims, nseqs, self.hidden_size).contiguous()  # bidirectional
        out, _ = self.gru(x,h_init)
        if self.bidirectional:
            forward_out = out[:, :, :self.hidden_size]
            backward_out = out[:, :, self.hidden_size:]
            out = forward_out + backward_out
            #out = torch.cat((forward_out, backward_out), dim=2)


        out = self.fc(out)

        return out

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = seq_len
        self.bidirectional = False
        self.num_layers = 1
        self.init_dims = 2 * self.num_layers if self.bidirectional else 1 * self.num_layers
        #self.gru = nn.GRU(1, self.hidden_size, batch_first=True,bidirectional=self.bidirectional)
        #self.gru = manualGRU(1,self.hidden_size,self.max_len)
        self.gru = miniGRU(1, self.hidden_size, self.max_len) #todo: think how to fox the mini gru, something is off
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float().unsqueeze(-1)  # (B, T, 1)
        out, _ = self.gru(x)

        if self.bidirectional:
            forward_out = out[:, :, :self.hidden_size]
            backward_out = out[:, :, self.hidden_size:]
            out = forward_out + backward_out
        out = self.fc(out)
        out = out.squeeze(-1)
        return out #[n,l, 1]

class ToyDatasets():
    def __init__(self, dataset_name,n_samples,seq_len):

        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.seq_len = seq_len

    def generate_dataset(self):
        """Sequential dataset to test if the model can guess the next number"""
        X = []
        Y = []

        start_idxs = np.random.randint(0,50,(self.n_samples))
        end_idxs = start_idxs +self.seq_len + 1

        for start_idx, end_idx in zip(start_idxs,end_idxs):
            seq = torch.arange(start_idx,end_idx)
            X.append(seq[:-1])
            Y.append(seq[1:])

        return torch.stack(X), torch.stack(Y)

    def non_linear_recurrency(self):

        x = np.zeros(self.n_samples)
        x[0], x[1] = 1, 1
        for i in range(2, self.n_samples):
            x[i] = x[i - 1] + x[i - 2] + 0.1 * x[i - 1] * x[i - 2]
        return x

    def lucas_sequences_dataset(self,steps: int=1):

        """
        Generate a batch of Lucas-like sequences. Harder case to understand if the rnn has only learnt patterns or also some sum rules

        Args:
            n_sequences (int): number of sequences to generate
            seq_len (int): length of each sequence
            random_init (bool): if True, randomize first two values
            normalize (bool): if True, normalize each sequence

        Returns:
            np.ndarray: shape (n_sequences, seq_len)
        """
        normalize = False
        sequences = torch.zeros((3, seq_len))

        start_idxs = [2,6,9] #be careful so that the final number is in the dataset
        end_idxs = [1,3,5]

        for i,(start,end) in enumerate(zip(start_idxs,end_idxs)):

            sequences[i, 0] = start
            sequences[i, 1] = end

            for t in range(2, seq_len):
                sequences[i, t] = sequences[i, t - 1] + sequences[i, t - 2]

            if normalize:
                sequences[i] = sequences[i] / torch.max(torch.abs(sequences[i]))


        true_prediction = sequences[:,1:]

        for _ in range(steps):
            last_val = true_prediction[:,-2] + true_prediction[:,-1]
            true_prediction = torch.concat([true_prediction,last_val[:,None]],dim=1)

        return sequences.long(), true_prediction.long()

    def fibonacci_dataset(self, vocab:int=20):
        """Fibonacci dataset to test if the model can learn the rule of sum, the initial 2 numbers go from 0 to vocab number"""
        X = torch.zeros(self.n_samples, self.seq_len, dtype=torch.long)
        Y = torch.zeros(self.n_samples, self.seq_len, dtype=torch.long)

        for i in range(self.n_samples):
            a = torch.randint(0, vocab, (1,))
            b = torch.randint(0, vocab, (1,))
            seq = list(sorted([a.item(), b.item()]))


            for _ in range(seq_len-1):
                seq.append((seq[-2] + seq[-1]))
            seq = torch.tensor(seq)
            X[i] = seq[:-1]
            Y[i] = seq[1:]

        return X, Y

    def build(self):
        if self.dataset_name == "sequential":
            return self.generate_dataset()
        elif self.dataset_name == "fibonacci":
            return self.fibonacci_dataset()
        else:
            raise ValueError(f"option {self.dataset_name} not available")



seq_len = 6
n_seqs = 10000
batch_size = 100
hidden_size = 10  #tried 64
embedding_size = 5
vocab_size = 50 + seq_len + 1 #should be the max number of numbers that we have, so that is 50 + seq_len + 1


# np.random.seed(0)
# torch.manual_seed(0)
x, y = ToyDatasets("fibonacci",n_seqs,seq_len).build()
vocab_size = max(x.max(),y.max()) + 1
print(f"vocabulary size is {vocab_size}")
dataset = torch.utils.data.TensorDataset(x,y) #unsqueeze(-1) if there is no embeddings
train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)




nepochs = 150


train_embeddings =False

if train_embeddings :
    model = GRUModel1()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_acc = 0
    total_loss = 0
    total = 0
    for epoch in range(nepochs):
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
            epoch_acc = (preds == y).float().mean()

            total_loss += loss.item()
            total_acc += epoch_acc.item()
            total += y.numel() #batchsize * seqlen

            print(
                f"epoch {epoch} "
                f"epoch loss {round(loss.item(),3)} "
                f"epoch acc {round(epoch_acc.item(), 3)} "
            )


else:
    model = GRUModel()
    criterion =  torch.nn.MSELoss() #squared l2 norm
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_acc = 0
    total_loss = 0
    total = 0
    for epoch in range(nepochs):

        tepoch = tqdm(train_loader, unit="batch",desc=f'Epoch {epoch}',leave=False if epoch < nepochs -1 else True)

        for x, y in tepoch:

            logits = model(x) #logits should approximate the true integers
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_acc = (torch.round(logits) == y).float().mean()
            mae = torch.abs(logits - y).mean()

            total_loss += loss.item()
            total_acc += epoch_acc.item()
            total += y.numel() #batchsize * seqlen

            # print(
            #     f"epoch {epoch} "
            #     f"epoch loss {round(loss.item(),3)} "
            #     f"epoch acc {round(epoch_acc.item(), 3)} "
            # )
            tepoch.set_postfix(epoch_loss=round(loss.item(),3), epoch_accuracy=100. * round(epoch_acc.item(), 3))


model.eval()

nsteps = 3
#test_input, true_pred = torch.tensor([[20,21,22,23,24,25]]), torch.tensor([[21,22,23,24,25,26]])  # input,for the mini GRU, the train and test sequences need to have the same max_len
#test_input, true_pred = torch.tensor([[12,13,14,15,16,17]]), torch.tensor([[13,14,15,16,17,18]])  # input
#test_input, true_pred = torch.tensor([[6,9,15,24,39,63]]), torch.tensor([[9,15,24,39,63,102]])
test_input, true_pred = ToyDatasets("fibonacci",1,seq_len).lucas_sequences_dataset(steps=1)
test_input_generative, true_pred_generative = ToyDatasets("fibonacci",1,seq_len).lucas_sequences_dataset(steps=3)
assert test_input.shape[1] == seq_len, f"train and test must have same max seq length, got {test_input.shape[1]}, expected {seq_len}"


def generate_sequence(model, start_seq, steps):
    """
    start_seq: (1, T) tensor, e.g. [[2, 1]]
    steps: number of future steps to generate
    """
    model.eval()

    x = start_seq.clone().float()  # (1, T)

    preds = []

    for _ in range(steps):
        print(f"generating step {_}")
        # forward pass
        out = model(x)  # (1, T)
        # store only last predicted token
        next_val = out[:, -1:]  # (1, 1) #the : keeps the dimensionality, it is equal to out[:,-1][None,:]
        # store prediction
        preds.append(next_val)

        # append prediction to sequence to extend it
        x = torch.cat([x, next_val], dim=1)

    return torch.cat(preds, dim=1)


with torch.no_grad():
    logits = model(test_input)

    if train_embeddings:
        preds_1step = logits.argmax(dim=-1)
    else:
        preds_1step = torch.round(logits)

    preds_nsteps = torch.round(generate_sequence(model,test_input,steps=3)) #todo: how to in the miniGru?

print("Input: ", test_input) #for each token in the input, we predict the next one [20->21, 21-> 22, ...]
print("Predicted/Generated sequence 1 step/token:", preds_1step) #should be [21,22,23,24,25]
print("Expected:", true_pred)
accuracy = (preds_1step == true_pred).float().mean(axis=1)
print(f"Accuracy per seq: {accuracy}, accuracy total: {accuracy.mean()}")
print("---------------------------------------------------------------------------")
print(f"Generated predictions n-steps: {preds_nsteps}")
print(f"Expected predictions n-steps: {true_pred_generative[:,seq_len-1:]}")

accuracy_gen = (preds_nsteps == true_pred_generative[:,seq_len-1:]).float().mean(axis=1)

print(f"Accuracy (generative) per seq: {accuracy_gen}, accuracy (generative) total: {accuracy_gen.mean()}")





