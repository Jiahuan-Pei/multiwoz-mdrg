from __future__ import division, print_function, unicode_literals

import argparse
import json
import random
import time
from io import open

import numpy as np
import torch
from torch.optim import Adam

from utils import util
from model.model import Model

# pp added: print out env
util.get_env_info()

parser = argparse.ArgumentParser(description='multiwoz-bsl-tr')
# Group args
# 1. Data & Dirs
data_arg = parser.add_argument_group(title='Data')
data_arg.add_argument('--data_dir', type=str, default='data', help='the root directory of data')
data_arg.add_argument('--log_dir', type=str, default='logs')
data_arg.add_argument('--model_dir', type=str, default='results/bsl_g/model/')
data_arg.add_argument('--model_name', type=str, default='translate.ckpt')
data_arg.add_argument('--train_output', type=str, default='data/train_dials/', help='Training output dir path')

# 2.Network
net_arg = parser.add_argument_group(title='Network')
net_arg.add_argument('--cell_type', type=str, default='lstm')
net_arg.add_argument('--attention_type', type=str, default='bahdanau')
net_arg.add_argument('--depth', type=int, default=1, help='depth of rnn')
net_arg.add_argument('--emb_size', type=int, default=50)
net_arg.add_argument('--hid_size_enc', type=int, default=150)
net_arg.add_argument('--hid_size_dec', type=int, default=150)
net_arg.add_argument('--hid_size_pol', type=int, default=150)
net_arg.add_argument('--max_len', type=int, default=50)
net_arg.add_argument('--vocab_size', type=int, default=400, metavar='V')
net_arg.add_argument('--use_attn', type=util.str2bool, nargs='?', const=True, default=True) # F
net_arg.add_argument('--use_emb',  type=util.str2bool, nargs='?', const=True, default=False)

# 3.Train
train_arg = parser.add_argument_group(title='Train')
train_arg.add_argument('--mode', type=str, default='train', help='training or testing: test, train, RL')
train_arg.add_argument('--optim', type=str, default='adam')
train_arg.add_argument('--max_epochs', type=int, default=20) # 15
train_arg.add_argument('--lr_rate', type=float, default=0.005)
train_arg.add_argument('--lr_decay', type=float, default=0.0)
train_arg.add_argument('--l2_norm', type=float, default=0.00001)
train_arg.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')
train_arg.add_argument('--teacher_ratio', type=float, default=1.0, help='probability of using targets for learning')
train_arg.add_argument('--dropout', type=float, default=0.0)
train_arg.add_argument('--early_stop_count', type=int, default=2)
train_arg.add_argument('--epoch_load', type=int, default=0)
train_arg.add_argument('--load_param', type=util.str2bool, nargs='?', const=True, default=False)


# 4. MISC
misc_arg = parser.add_argument_group('MISC')
misc_arg.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
misc_arg.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
misc_arg.add_argument('--db_size', type=int, default=30)
misc_arg.add_argument('--bs_size', type=int, default=94)
misc_arg.add_argument('--no_cuda',  type=util.str2bool, nargs='?', const=True, default=False)

# 5. Here add new args

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('args.cuda={}'.format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
util.init_seed(args.seed)


def train(print_loss_total,print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor, name=None):
    # create an empty matrix with padding tokens
    input_tensor, input_lengths = util.padSequence(input_tensor)
    target_tensor, target_lengths = util.padSequence(target_tensor)
    bs_tensor = torch.as_tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.as_tensor(db_tensor, dtype=torch.float, device=device)

    # pp added -- start
    data = input_tensor, target_tensor, bs_tensor, db_tensor
    if torch.cuda.is_available():
        data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in
                range(len(data))]
    input_tensor, target_tensor, bs_tensor, db_tensor = data
    # pp added -- end

    loss, loss_acts, grad = model.train(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor,
                             bs_tensor, name)

    #print(loss, loss_acts)
    print_loss_total += loss
    print_act_total += loss_acts
    print_grad_total += grad

    model.global_step += 1
    model.sup_loss = torch.zeros(1)

    return print_loss_total, print_act_total, print_grad_total


def trainIters(model, n_epochs=10, args=args):
    prev_min_loss, early_stop_count = 1 << 30, args.early_stop_count
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        print_loss_total = 0; print_grad_total = 0; print_act_total = 0  # Reset every print_every
        start_time = time.time()
        # watch out where do you put it
        model.optimizer = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.parameters()), weight_decay=args.l2_norm)
        model.optimizer_policy = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.policy.parameters()), weight_decay=args.l2_norm)

        dials = train_dials.keys()
        random.shuffle(dials)
        input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = []
        for name in dials:
            val_file = train_dials[name]
            model.optimizer.zero_grad()
            model.optimizer_policy.zero_grad()

            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)

            if len(db_tensor) > args.batch_size:
                print_loss_total, print_act_total, print_grad_total = train(print_loss_total, print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor)
                input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = [];

        print_loss_avg = print_loss_total / len(train_dials)
        print_act_total_avg = print_act_total / len(train_dials)
        print_grad_avg = print_grad_total / len(train_dials)
        print('TIME:', time.time() - start_time)
        print('Time since %s (Epoch:%d %d%%) Loss: %.4f, Loss act: %.4f, Grad: %.4f' % (util.timeSince(start, epoch / n_epochs),
                                                            epoch, epoch / n_epochs * 100, print_loss_avg, print_act_total_avg, print_grad_avg))

        # VALIDATION
        valid_loss = 0
        for name, val_file in val_dials.items():
            input_tensor = []; target_tensor = []; bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor,
                                                                                         target_tensor, bs_tensor,
                                                                                         db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.as_tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.as_tensor(db_tensor, dtype=torch.float, device=device)

            # pp added -- start
            data = input_tensor, target_tensor, bs_tensor, db_tensor
            if torch.cuda.is_available():
                data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in
                        range(len(data))]
            input_tensor, target_tensor, bs_tensor, db_tensor = data
            # pp added -- end

            proba, _, _ = model.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)
            proba = proba.view(-1, model.vocab_size) # flatten all predictions
            loss = model.gen_criterion(proba, target_tensor.view(-1))
            valid_loss += loss.item()


        valid_loss /= len(val_dials)
        print('Current Valid LOSS:', valid_loss)

        model.saveModel(epoch)


if __name__ == '__main__':
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=args.data_dir)
    # Load training file list:
    with open('{}/train_dials.json'.format(args.data_dir)) as outfile:
        train_dials = json.load(outfile)

    # Load validation file list:
    with open('{}/val_dials.json'.format(args.data_dir)) as outfile:
        val_dials = json.load(outfile)

    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(args.epoch_load)

    trainIters(model, n_epochs=args.max_epochs, args=args)
