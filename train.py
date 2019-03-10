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

rev_vocab_path = './AuxData/rev_vocab'

def train(input_tensor, target_tensor, target_lengths, encoder, decoder, criterion, optimizer, device, need_grad=True, teacher_forcing_ratio=0):

    if need_grad:
        optimizer.zero_grad()

    loss = 0
    maxlen = max(30, target_tensor.shape[1])
    # Get encoder hidden states and outputs
    output_e, hidden_e = encoder(input_tensor)

    output_d, hidden_d = decoder(target_tensor[:,:-1], hidden_e, maxlen, teacher_forcing_ratio = teacher_forcing_ratio)
    # Define the loss function
    batch_size = output_d.shape[0]
    for i in range(batch_size):
        loss += criterion(output_d[i,:target_lengths[i]-1,:], target_tensor[i,1:target_lengths[i]])
    
    loss /= batch_size
    #####################################

    if need_grad:
        loss.backward()

        optimizer.step()

    return loss.item(), output_d

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

def train_per_epoch(loader, encoder, decoder, criterion, optimizer, device, need_grad=True, teacher_forcing_ratio=0):
    ld = iter(loader)

    numIters = len(ld)
    qdar = tqdm.tqdm(range(numIters),
                            total= numIters,
                            ascii=True)
    n = 0
    loss = 0
    for itr in qdar: 
        inputs = makeInp(next(ld))
        input_tensor = inputs['question']
        target_tensor = inputs['response']
        target_length = inputs['rLengths']
        loss_step, out = train(input_tensor, target_tensor, target_length, encoder, decoder, criterion, optimizer, device, need_grad=need_grad, teacher_forcing_ratio=teacher_forcing_ratio)
        loss += loss_step
        n += 1
    loss /= n
    return loss

def rev_vocab(idx_seq):
    with open(rev_vocab_path),"rb") as fp:
        rev_vocab = pickle.load(fp,encoding='latin1')
    
    sentence_out = []
    for idx in idx_seq:
        sentence_out.append(rev_vocab[idx])
        # stop at the first __eou__
        if idx == 3:
            break
    
    with open('result', 'a') as f:
        f.write(' '.join(sentence_out))
        f.write('\n')

def evaluate(loader, encoder, decoder):
    ld = iter(loader)

    numIters = len(ld)
    qdar = tqdm.tqdm(range(numIters),
                            total= numIters,
                            ascii=True)
    n = 0
    loss = 0
    for itr in qdar: 
        inputs = makeInp(next(ld))
        input_tensor = inputs['question']
        target_tensor = inputs['response']
        target_length = inputs['rLengths']
        loss_step, out = train(input_tensor, target_tensor, target_length, encoder, decoder, criterion, optimizer, device, need_grad=need_grad, teacher_forcing_ratio=teacher_forcing_ratio)
        loss += loss_step
        
        # batch size for the test data is 1, but we loop through all batches here anyway
        for i in range(out.shape[0]):
            sentence = []
            for j in range(out.shape[1]):
                word = torch.topk(out[i,j,:], 1)[1].squeeze(1)
                sentence.append(word)
            rev_vocab(sentence)
        n += 1
    loss /= n
    return loss


def trainIters(loader, encoder, decoder, max_epoch, device, learning_rate=0.01):
    start = time.time()
    plot_losses = []

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.SGD(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_dev_loss = None
    for epoch in range(max_epoch):
        loss = train_per_epoch(loader.ldTrain, encoder, decoder, criterion, optimizer, device, teacher_forcing_ratio=0.5)
        print('Epoch '+str(epoch)+': perplexity on the train set: '+str(math.exp(loss)))
        with torch.no_grad():
            dev_loss = train_per_epoch(loader.ldDev, encoder, decoder,criterion, optimizer, device, need_grad=False)
            print('perplexity on the dev set: '+str(math.exp(dev_loss)))
            # # save the best model
            # if best_dev_loss is None or best_dev_loss > dev_loss:
            #     best_dev_loss = dev_loss
            #     torch.save(encoder.state_dict(), PATH)
    


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

    trainIters(loader, encoder, decoder, 100, device, learning_rate=0.01)

if __name__ == "__main__":
    main()
