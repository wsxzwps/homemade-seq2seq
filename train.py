from model import *
from loader import *
from utils import ConfigParser, makeInp

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

import tqdm

def train(input_tensor, target_tensor, target_lengths, encoder, decoder, criterion, optimizer, device):

    optimizer.zero_grad()

    loss = 0

    # Get encoder hidden states and outputs
    output_e, hidden_e = encoder(input_tensor)

    output_d, hidden_d = decoder(target_tensor[:,:-1], hidden_e)
    # Define the loss function
    batch_size = output_d.shape[0]
    for i in range(batch_size):
        loss += criterion(output_d[i,:target_lengths[i]-1,:], target_tensor[i,1:target_lengths[i]])
    
    loss /= batch_size
    #####################################

    loss.backward()

    optimizer.step()

    return loss.item()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(loader, encoder, decoder, n_iters, device, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.SGD(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    ld = iter(loader.ldTrain)

    numIters = len(ld)
    qdar = tqdm.tqdm(range(numIters),
                            total= numIters,
                            ascii=True)
    for itr in qdar: 
        inputs = makeInp(next(ld))
        input_tensor = inputs['question']
        target_tensor = inputs['response']
        target_length = inputs['rLengths']
        loss = train(input_tensor, target_tensor, target_length, encoder, decoder, criterion, optimizer, device)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ConfigParser.parse_config()
    
    loader = LoaderHandler(config)

    embedding_path = config['model']['embedding']
    embedding = torch.FloatTensor(np.load(embedding_path))
    vocab_size = len(embedding)


    hidden_size = config['model']['hidden_size']
    batch_size = config['loader']['batchSize']
    encoder = EncoderRNN(vocab_size, 100, hidden_size, batch_size, embedding).to(device)
    decoder = DecoderRNN(vocab_size, 100, hidden_size, batch_size, embedding).to(device)

    trainIters(loader, encoder, decoder, 1000, device, print_every=100)

if __name__ == "__main__":
    main()
