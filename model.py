import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import random

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size, embedding=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

        self.gru = nn.GRU(embedding_size, hidden_size, batch_first = True)

    
    def forward(self, input):
        embs = self.embedding(input)
        output, hidden = self.gru(embs)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size, embedding=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.sos_id = 2

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, input, hidden, max_len, teacher_forcing_ratio=1):
        if random.random() < teacher_forcing_ratio:
            embs = self.embedding(input)
            output, hidden = self.gru(embs, hidden)
            output = F.log_softmax(self.out(output), dim=1)
        else:
            words = torch.zeros([input.shape[0],1]) + self.sos_id
            words = torch.LongTensor(words)
            out = torch.zeros(input.shape[0], max_len,self.embedding_size)
            for i in range(max_len):
                embs = self.embedding(words)
                output, hidden = self.gru(embs, hidden)
                scores = F.log_softmax(self.out(output), dim=1)
                out[:,i,:] = scores
                for j in range(input.shape[0]):
                    words[j] = torch.argmax(scores[j])

        return output, hidden

