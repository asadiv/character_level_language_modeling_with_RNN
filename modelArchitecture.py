import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1) # to add a dimesnsion for seq length (b,seq,embedings)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        # convert the (batch,1,vocab)to (batch,vocab) as CEloss would expect
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell
    # to initiate empty hidden and call state for the first char of a seq
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size) # 1 cuz only 1 layer in lstm
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(device), cell.to(device)  # torch.zeros is created on cpu