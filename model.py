import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        self.gru = nn.GRU(embedding_size, hidden_size)

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def forward(self, input, hidden):
        embs = self.embedding(input).view(len(input), 1, -1)
        output, hidden = self.gru(embs, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def forward(self, input, hidden):
        embs = self.embedding(input).view(len(input), 1, -1)
        output, hidden = self.gru(embs, hidden)
        output = output.view(len(input), -1)
        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden

