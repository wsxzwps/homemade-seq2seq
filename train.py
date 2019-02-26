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

def train(input_tensor, target_tensor, encoder, decoder, criterion, optimizer, device):
    encoder_hidden = encoder.initHidden(device)

    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    # Get encoder hidden states and outputs
    output_e, hidden_e = encoder(input_tensor, encoder_hidden)
    
    # Initialize decoder hidden state

    
    # Get decoder hidden states and outputs
    output_d, hidden_d = decoder(target_tensor, hidden_e)
    # Define the loss function
    loss = criterion(output_e, target_tensor)
    
    #####################################

    loss.backward()

    optimizer.step()

    return loss.item() / target_length

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
    criterion = nn.NLLLoss()

    ld = iter(loader.ldTrain)

    numIters = len(ld)
    qdar = tqdm.tqdm(range(numIters),
                            total= numIters,
                            ascii=True)
    for itr in qdar: 
        inputs = makeInp(next(ld))
        input_tensor = inputs['question']
        target_tensor = inputs['response']
        loss = train(input_tensor, target_tensor, encoder, decoder, criterion, optimizer, device)
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
    encoder = EncoderRNN(vocab_size, 100, hidden_size, embedding).to(device)
    decoder = DecoderRNN(vocab_size, 100, hidden_size, embedding).to(device)

    trainIters(loader, encoder, decoder, 10000, device, print_every=1000)

if __name__ == "__main__":
    main()
