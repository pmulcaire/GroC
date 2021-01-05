#    This code builds on the AWD-LSTM codebase
#    (https://github.com/salesforce/awd-lstm-lm).
#
#    groc is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    groc is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with groc. If not, see http://www.gnu.org/licenses/

import argparse
import os, shutil
import hashlib
import time
import math
import numpy as np
import torch
import torch.nn as nn
import data
import model
from utils import batchify, get_batch, repackage_hidden, get_external_knowledge
import sys
import random
import pickle

from collections import deque

import IPython as ipy

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--test_data', type=str, default='data/penn/',
                    help='location of the test data corpus')
parser.add_argument('--save', type=str,
                    help='path from which to load the model')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='use CUDA device')
parser.add_argument('--cuda_device', type=int, default=-1,
                    help='set CUDA device')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--test_batch_size', type=int, default=64,
                    help='test batch size')
parser.add_argument('--adapt_method', default="change_vocab",
                    help='method to adapt to new vocabulary')
parser.add_argument('--hyp_search', type=str, default=None,
                    help='search over ranges for various hyperparams (edit file to specify values)')
parser.add_argument('--downweight_oov', type=float, default=-1.0,
                    help='weight for new words in test vocab')
parser.add_argument('--langs', default=None,
                    help='languages to use from test corpus')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda and --cuda_device [device_id]")
    else:
        torch.cuda.set_device(int(args.cuda_device))
        torch.cuda.manual_seed(args.seed)

def logging(s, print_=True, log_=True):
    print(s, file=sys.stderr)
    if log_:
        with open(os.path.join(args.save, 'eval_log.txt'), 'a+') as f_log:
            f_log.write(str(s) + '\n')


def model_load(fn, device=0):
    with open(fn+'/model.pt', 'rb') as f:
        model = torch.load(f, map_location=f'cuda:{device}')
        if 'langs' not in model.H.__dict__:
            model.H.langs = None
            model.H.max_charlen = 20
            #ipy.embed()
    with open(fn+'/criterion.pt', 'rb') as f:
        criterion = torch.load(f, map_location=f'cuda:{device}')
    with open(fn+'/optimizer.pt', 'rb') as f:
        optimizer = torch.load(f, map_location=f'cuda:{device}')
    return model, criterion, optimizer

def corpus_load(corpus_path, use_unk=False):
    langs = '-'.join(args.langs.split(','))
    fn = 'corpus.{}.data'.format(hashlib.md5((corpus_path.strip('/')+langs+"-test").encode()).hexdigest())
    print (fn)
    if os.path.exists(fn):
        logging('Loading cached {} test dataset from {}...'.format(langs, corpus_path))
        corpus = torch.load(fn)
    else:
        logging('Producing {} test dataset from {} ...'.format(langs, corpus_path))
        if args.langs is None:
            corpus = data.Corpus(args.test_data, use_unk=use_unk)
        else:
            corpus = data.MultilingualCorpus(args.test_data, args.langs, use_unk=use_unk)
        torch.save(corpus, fn)
    return corpus

def evaluate(model, criterion, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n = data_source.size(0)
    output_dim = model.H.emsize
    V = len(model.dict.word2idx)
    batch_row_idx = torch.arange(batch_size).long()

    hidden = model.init_hidden(batch_size)
    if not os.path.isfile(os.path.join(args.save, 'recover-state.pkl')):
        start_iter = 0
        total_loss = 0
    else:
        logging("Restoring from recover-state.pkl...")
        with open(os.path.join(args.save, 'recover-state.pkl'),'rb') as f:
            start_iter, total_loss = pickle.load(f)
        logging("Restoring from recover-cache-targets.pt...")
        if len(cache) > 0:
            print("{} {}".format(cache[0][0].size(), cache[0][1].size()))
        logging("Restore complete.")

    if isinstance(criterion, nn.CrossEntropyLoss):
        softmax = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()

    if args.downweight_oov > 0.0 and args.downweight_oov < 1.0:
        dw_inds = model.get_new()

    for i in range(start_iter, n-1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden, eval_mode=True)

        if isinstance(criterion, nn.CrossEntropyLoss):
            logits = torch.mm(output,weight.t())
            if args.downweight_oov > 0.0 and args.downweight_oov < 1.0:
                lexp = logits.exp()
                lexp[:,dw_inds] *= args.downweight_oov
                logits = lexp.log()
            logits += bias
            if args.adapt_method == "change_vocab":
                loss = criterion(logits, targets).data
            else:
                raise ValueError("Unknown adaptation method: {}".format(args.adapt_method))
            total_loss += len(data) * loss
        hidden = repackage_hidden(hidden)

        print("\r{}/{} - ppl: {:3.2f}".format(i,n, math.exp(total_loss/n) ))

        if i % 100 == 0 and i > 0:
            logging("{}/{} - ppl: {:3.2f}".format(i,n, math.exp(total_loss/n)))
        if i % 5000 == 0 and i > 0:
            logging("Saving to recover-state.pkl...")
            with open(os.path.join(args.save, 'recover-state.pkl'),'wb') as f:
                pickle.dump((i, total_loss),f)
            logging("Save complete.")

    return total_loss.item() / n

# Log command
logging("Command: python " + " ".join(sys.argv))

# Load the best saved model.
model, criterion, optimizer = model_load(args.save, device=args.cuda_device)

if args.adapt_method in ["change_vocab"]:
    """
    use the original vocab for indexing, no change to model parameters
    K = size of new words (OOVs in the original vocab)
    N = size of train vocab plus new words
    uniform interpolation:
    - likelihood = (lambda * likelihood) + ((1-lambda) * 1/N)
    unigram interpolation:
    - likelihood = (lambda * likelihood) +
                   ((1-lambda) * (p_unigram(target)/sum(p_unigram(w) for w in observed_vocab)))
    neural cache interpolation:
    - likelihood = (lambda * likelihood) +
                   ((1-lambda) * (p_cache(target)/sum(p_cache(w) for w in cache_vocab)))
    """
    test_corpus = corpus_load(args.test_data)
    model.H.ntoken = len(test_corpus.dictionary.idx2word)
    char_arr, rel_arr, def_arr = get_external_knowledge(model.H, test_corpus)
    model.change_embedding_vocab(char_arr, rel_arr, def_arr,
                                 test_corpus.dictionary, set_zero=True)
    logging("Vocab size pre-change: {}".format(len(model.old_dict.word2idx)))
    logging("Vocab size post-change: {}".format(len(model.dict.word2idx)))
else:
    raise AssertionError("new vocabulary provided but model vocab not changed")

test_data = batchify(test_corpus.test, args.test_batch_size, args)

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
else:
    model = model.cpu()
    criterion = criterion.cpu()

# Run on test data.
logging("Evaluating...")
with torch.no_grad():
    test_loss = evaluate(model, criterion, test_data, args.test_batch_size)
    print("")
    logging('=' * 89)
    logging('| End of evaluation | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
    logging
    logging('=' * 89)
