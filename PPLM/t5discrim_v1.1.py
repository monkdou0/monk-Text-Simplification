#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import sys
import argparse
from tqdm import trange
from torchtext import data as torchtext_data
from torchtext import datasets

import torch
import torch.utils.data as data

from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
from operator import add
#from run_gpt2 import top_k_logits
from style_utils import to_var
import copy
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained('/home/zhangming/simplification/t5trainModel/t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('/home/zhangming/simplification/t5GetRes/monkmodelSumm')

torch.manual_seed(0)
np.random.seed(0)
device='cuda:1'
ESIZE=768
#lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
#sys.path.insert(1, lab_root)

#from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from torch.autograd import Variable



class ClassificationHead(torch.nn.Module):

    def __init__(self, class_size=2, embed_size=ESIZE):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = (torch.nn.Linear(embed_size, class_size))

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        # lm_logits = F.relu(self.mlp1(hidden_state))
        # lm_logits = self.mlp2(lm_logits)
        lm_logits = self.mlp(hidden_state)
        return lm_logits


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifierhead = ClassificationHead()
        self.model = model
        self.spltoken = Variable(torch.randn(1, 1, 1024).type(torch.FloatTensor), requires_grad=True)
        self.spltoken = self.spltoken.repeat(10, 1, 1)
        self.spltoken = self.spltoken.cuda()

    def train(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass

    def forward(self, x):
        x = model.forward_embed(x)
        x = torch.cat((x, self.spltoken), dim=1)
        _, x = model.forward_transformer_embed(x, add_one=True)
        x = self.classifierhead(x[-1][:, -1, :])
        x = F.log_softmax(x, dim=-1)
        return x


class Discriminator2(torch.nn.Module):
    def __init__(self, class_size=5, embed_size=ESIZE):
        super(Discriminator2, self).__init__()
        self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        self.model = model
        self.embed_size = embed_size

    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def forward(self, x):
        x = model.forward_embed(x)
        hidden, x = model.forward_transformer_embed(x)
        x = torch.sum(hidden, dim=1)
        x = self.classifierhead(x)
        x = F.log_softmax(x, dim=-1)
        return x

def model_output(model,input_ids,output_so_far,past_key_values=None):
    o=model(input_ids=input_ids,decoder_input_ids=output_so_far,past_key_values=past_key_values,return_dict=True,output_hidden_states=True)
    return o.logits,o.past_key_values,o.decoder_hidden_states


class Discriminator2mean(torch.nn.Module):
    def __init__(self, class_size=2, embed_size=ESIZE):
        super(Discriminator2mean, self).__init__()
        self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        self.model = model
        self.embed_size = embed_size

    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def forward(self, encode_x,x):
        mask_src = 1 - x.eq(0).unsqueeze(1).type(torch.FloatTensor).detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1)
        mask_src=mask_src.to(device)
        _,_,hidden= model_output(model,encode_x,x)
        hidden=hidden[-1]
        #  Hidden has shape batch_size x length x embed-dim

        hidden = hidden.permute(0, 2, 1)
        _, _, batch_length = hidden.shape
        hidden = hidden * mask_src  # / torch.sum(mask_src, dim=-1).unsqueeze(2).repeat(1, 1, batch_length)
        #
        hidden = hidden.permute(0, 2, 1)
        x = torch.sum(hidden, dim=1)/(torch.sum(mask_src, dim=-1).detach() + 1e-10)
        x = self.classifierhead(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        d = {}
        d['X'] = self.X[index]
        d['y'] = self.y[index]
        return d


def collate_fn(data):
    def merge(sequences,idx):
        lengths = [len(seq[idx]) for seq in sequences]

        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()  # padding index 0
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[idx][:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["X"][1]), reverse=True)  # sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # input
    encode_x_batch, _ = merge(item_info['X'],0)
    decode_x_batch, _ = merge(item_info['X'],1)
    y_batch = item_info['y']

    return encode_x_batch, decode_x_batch, torch.tensor(y_batch, device=device, dtype=torch.long)


def train_epoch(data_loader, discriminator, device='cuda', args=None, epoch=1):
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator.train_custom()

    for batch_idx, (encode_data,data, target) in enumerate(data_loader):
        encode_data,decode_data, target = encode_data.to(device),data.to(device), target.to(device)

        optimizer.zero_grad()

        output = discriminator(encode_data,decode_data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Relu Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(data_loader, discriminator, device='cuda', args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for encode_data,data, target in data_loader:
            encode_data,data, target = encode_data.to(device),data.to(device), target.to(device)
            output = discriminator(encode_data,data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='Train a discriminator on top of GPT-2 representations')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('--save-model', action='store_false', help='whether to save the model')
    parser.add_argument('--wikilarge_dir', default='../wikismall/data-simplification/wikilarge',type=str)
    parser.add_argument('--dataset_label', default='wikilarge',type=str)
    args = parser.parse_args()

    batch_size = args.batch_size
    # load sst
    if True:
        COMPLEX=0
        SIMPLE=1
        cx = []
        sx = []
        with open(os.path.join(args.wikilarge_dir,'wiki.full.aner.ori.train.src')) as f:
            for d in f.readlines():
                try:
                    seq=tokenizer.encode(d)
                    seq=torch.tensor(seq,device=device,dtype=torch.long)
                    cx.append(seq)
                except:
                    continue
        with open(os.path.join(args.wikilarge_dir,'wiki.full.aner.ori.train.dst')) as f:
            for d in f.readlines():
                try:
                    seq=tokenizer.encode(d)
                    seq=torch.tensor(seq,device=device,dtype=torch.long)
                    sx.append(seq)
                except:
                    continue
        x=[]
        y=[]
        for idx in range(len(cx)):
            x.append([cx[idx],sx[idx]])
            y.append(SIMPLE)
            x.append([sx[idx],cx[idx]])
            y.append(COMPLEX)


        dataset = Dataset(x, y)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        discriminator = Discriminator2mean(class_size=2).to(device)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        train_epoch(discriminator=discriminator, data_loader=data_loader, args=args, device=device, epoch=epoch)
        test_epoch(data_loader=test_loader, discriminator=discriminator, args=args,device=device)

        if (args.save_model):
            torch.save(discriminator.state_dict(),
                       "discrim_models/{}_mean_lin_discriminator_{}.pt".format(args.dataset_label, epoch))
            torch.save(discriminator.get_classifier().state_dict(),
                       "discrim_models/{}_classifierhead.pt".format(args.dataset_label))



if __name__ == '__main__':
    main()
    

