# coding=utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import random
import time
from io import open

import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn

from utils import util, multiwoz_dataloader
from model.model import Model
from utils.util import detected_device, PAD_token
from model.evaluator import evaluateModel
# from tqdm import tqdm
# SOS_token = 0
# EOS_token = 1
# UNK_token = 2
# PAD_token = 3

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
data_arg.add_argument('--train_output', type=str, default='results/bsl_g/data/train_dials/', help='Training output dir path')

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
misc_arg.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
#
# 5. Here add new args
new_arg = parser.add_argument_group('New')
new_arg.add_argument('--intent_type', type=str, default=None, help='separate experts by intents: None, domain, sysact or domain_act') # pp added
# different implementation of moe
# 1. only weight loss & hyper weights
# --use_moe_loss=True --learn_loss_weight=False --use_moe_model=False
# 2. only weight loss & learn weights
# --use_moe_loss=True --learn_loss_weight=True --use_moe_model=False
# 3. only split model
# --use_moe_loss=False --learn_loss_weight=False --use_moe_model=True
# 4. both & hyper weights
# --use_moe_loss=True --learn_loss_weight=False --use_moe_model=True
# 5. both & learn weights
# --use_moe_loss=True --learn_loss_weight=True --use_moe_model=True
new_arg.add_argument('--use_moe_loss', type=util.str2bool, nargs='?', const=True, default=False, help='inner model weighting loss')
new_arg.add_argument('--learn_loss_weight', type=util.str2bool, nargs='?', const=True, default=False, help='learn weight of moe loss')
new_arg.add_argument('--use_moe_model', type=util.str2bool, nargs='?', const=True, default=False, help='inner model structure partition')
new_arg.add_argument('--debug', type=util.str2bool, nargs='?', const=True, default=False, help='if True use small data for debugging')
new_arg.add_argument('--train_valid', type=util.str2bool, nargs='?', const=True, default=False, help='if True add valid data for training')
new_arg.add_argument('--mu_expert', type=float, default=0.5)

args = parser.parse_args()
args.device = detected_device.type
print('args.device={}'.format(args.device))
print('args.intent_type={}'.format(args.intent_type))
# pp added: init seed
print(args)
util.init_seed(args.seed)

def eval_with_train3(model, val_dials, mode='Valid', policy='Greedy'):
    val_dials_gen = {0:{}, 1:{}}
    valid_loss = {0:0, 1:0}
    policy_idx_list = range(2) # both
    if policy == 'Greedy':
        policy_idx_list = [0]
    elif policy == 'Beam':
        policy_idx_list = [1]

    for name, val_file in val_dials.items():
        loader = multiwoz_dataloader.get_loader_by_dialogue(val_file, name,
                                                          input_lang_word2index, output_lang_word2index,
                                                          args.intent_type, intent2index)
        data = iter(loader).next()
        # Transfer to GPU
        if torch.cuda.is_available():
            data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
        input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data

        for ii in policy_idx_list:
            if ii == 0:
                model.beam_search = False
            else:
                model.beam_search = True
            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor, mask_tensor)

            valid_loss[ii] += loss_sentence
            val_dials_gen[ii][name] = output_words

    for ii in policy_idx_list:
        if ii == 0:
            model.beam_search = False
        else:
            model.beam_search = True
        BLEU, MATCHES, SUCCESS, SCORE, total = evaluateModel(val_dials_gen[ii], val_dials, delex_path, mode)
        print(50 * '-' + policy)
        print('{0} Loss:{1:.6f}'.format(mode, valid_loss[ii]))



def trainOne(print_loss_total,print_act_total, print_grad_total, input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor=None, name=None):

    loss, loss_acts, grad = model.model_train(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor, mask_tensor, name)
    # pp added: experts' loss
    # print('@'*20, '\n', target_tensor)
    if False and mask_tensor is not None:  # data separate by intents
        gen_loss_list = []
        if mask_tensor is not None:  # data separate by intents
            # print(mask_tensor)
            for mask in mask_tensor:  # each intent has a mask [Batch, 1]
                target_tensor_i = target_tensor.clone()
                target_tensor_i = target_tensor_i.masked_fill_(mask, value=PAD_token)
                # print(mask)
                # print(target_tensor_i)
                # print('*'*50)
                loss_i, loss_acts_i, grad_i = model.model_train(input_tensor, input_lengths, target_tensor_i, target_lengths, db_tensor, bs_tensor, mask_tensor, name)
                gen_loss_list.append(loss_i)
        # print('loss', loss, '; mean_experts_loss', torch.mean(torch.tensor(gen_loss_list)), '\ngen_loss_list', ['%.4f' % s if s!=0 else '0' for s in gen_loss_list])
        # loss = 0.5*loss + 0.5*torch.mean(torch.tensor(gen_loss_list))
        mu_expert = args.mu_expert
        loss = (1 - mu_expert) * loss + mu_expert * torch.mean(torch.tensor(gen_loss_list))
        # loss = 0.5*loss + 0.5*torch.mean(torch.tensor(gen_loss_list))
    #print(loss, loss_acts)
    print_loss_total += loss
    print_act_total += loss_acts
    print_grad_total += grad

    model.global_step += 1
    model.sup_loss = torch.zeros(1)

    return print_loss_total, print_act_total, print_grad_total


def trainIters(model, intent2index, n_epochs=10, args=args):
    prev_min_loss, early_stop_count = 1 << 30, args.early_stop_count
    start = time.time()
    # Valid_Scores, Test_Scores = [], []
    Scores = []
    for epoch in range(1, n_epochs + 1):
        print('%s\nEpoch=%s (%s %%)' % ('~'*50, epoch, epoch / n_epochs * 100))
        print_loss_total = 0; print_grad_total = 0; print_act_total = 0  # Reset every print_every
        start_time = time.time()
        # watch out where do you put it
        model.optimizer = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.parameters()), weight_decay=args.l2_norm)
        model.optimizer_policy = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.policy.parameters()), weight_decay=args.l2_norm)
        # Training
        model.train()
        step = 0
        for data in train_loader: # each element of data tuple has [batch_size] samples
            step += 1
            model.optimizer.zero_grad()
            model.optimizer_policy.zero_grad()
            # Transfer to GPU
            if torch.cuda.is_available():
                data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
            input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data
            print_loss_total, print_act_total, print_grad_total = trainOne(print_loss_total, print_act_total, print_grad_total, input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor)
            if step > 1 and args.debug:
                break # for debug
        train_len = len(train_loader) # 886 data # len(train_loader.dataset.datasets) # 8423 dialogues
        print_loss_avg = print_loss_total / train_len
        print_act_total_avg = print_act_total / train_len
        print_grad_avg = print_grad_total / train_len
        print('Train Time:%.4f' % (time.time() - start_time))
        print('Train Loss: %.6f\nTrain Grad: %.6f' % (print_loss_avg, print_grad_avg))

        if not args.debug:
            step = 0

        # VALIDATION
        if args.train_valid: # if add valid data for training
            model.train()
            valid_loss = 0
            for name, val_file in val_dials.items()[-step:]:
                loader = multiwoz_dataloader.get_loader_by_dialogue(val_file, name,
                                                                    input_lang_word2index, output_lang_word2index,
                                                                    args.intent_type, intent2index)
                data = iter(loader).next()
                # Transfer to GPU
                if torch.cuda.is_available():
                    data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
                input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data
                proba, _, _ = model.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor,
                                            bs_tensor, mask_tensor)  # pp added: mask_tensor
                proba = proba.view(-1, model.vocab_size)  # flatten all predictions
                loss = model.gen_criterion(proba, target_tensor.view(-1))
                valid_loss += loss.item()
            valid_len = len(val_dials) # 1000
            valid_loss /= valid_len
            # pp added: evaluate valid
            print('Train Valid Loss: %.6f' % valid_loss)

        # pp added
        model.eval()
        val_dials_gen = {}
        valid_loss = 0
        for name, val_file in val_dials.items()[-step:]:
            loader = multiwoz_dataloader.get_loader_by_dialogue(val_file, name,
                                                                input_lang_word2index, output_lang_word2index,
                                                                args.intent_type, intent2index)
            data = iter(loader).next()
            # Transfer to GPU
            if torch.cuda.is_available():
                data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
            input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data
            proba, _, _ = model.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor,
                                        bs_tensor, mask_tensor)  # pp added: mask_tensor
            proba = proba.view(-1, model.vocab_size)  # flatten all predictions
            loss = model.gen_criterion(proba, target_tensor.view(-1))
            valid_loss += loss.item()
            # pp added: evaluation - Plan A
            # model.eval()
            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor, mask_tensor)
            # model.train()
            val_dials_gen[name] = output_words
        valid_len = len(val_dials) # 1000
        valid_loss /= valid_len

        # pp added: evaluate valid
        print('Valid Loss: %.6f' % valid_loss)
        # BLEU, MATCHES, SUCCESS, SCORE, TOTAL
        Valid_Score = evaluateModel(val_dials_gen, val_dials, delex_path, mode='Valid')
        # Valid_Scores.append(Valid_Score)

        # Testing
        # pp added
        model.eval()
        test_dials_gen ={}
        for name, test_file in test_dials.items()[-step:]:
            loader = multiwoz_dataloader.get_loader_by_dialogue(test_file, name,
                                                                input_lang_word2index, output_lang_word2index,
                                                                args.intent_type, intent2index)
            data = iter(loader).next()
            # Transfer to GPU
            if torch.cuda.is_available():
                data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
            input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data
            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor, mask_tensor)
            test_dials_gen[name] = output_words
        # pp added: evaluate valid
        Test_Score = evaluateModel(test_dials_gen, test_dials, delex_path, mode='Test')
        # Test_Scores.append(Test_Score)

        model.train()
        # pp added: evaluation - Plan B
        # print(50 * '=' + 'Evaluating start...')
        # # eval_with_train(model)
        # eval_with_train3(model, val_dials, mode='valid')
        # eval_with_train3(model, test_dials, mode='test')
        # print(50 * '=' + 'Evaluating end...')

        model.saveModel(epoch)
        # BLEU, MATCHES, SUCCESS, SCORE, TOTAL
        Scores.append(tuple([epoch]) + Valid_Score + Test_Score) # combine the tuples; 11 elements
    import pandas as pd
    fields = ['Epoch', 'Valid BLEU', 'Valid Matches', 'Valid Success', 'Valid Score', 'Valid Dialogues',
              'Test BLEU', 'Test Matches', 'Test Success', 'Test Score', 'Test Dialogues']
    df = pd.DataFrame(Scores, columns=fields)
    sdf = df.sort_values(by=['Valid Score'], ascending=False)
    print('Top3:', '=' * 60)
    print(sdf.head(3).transpose())
    print('Best:', '=' * 60) # selected by valid score
    best_df = sdf.head(1)[['Test BLEU', 'Test Matches', 'Test Success', 'Test Score', 'Epoch']]
    print(best_df.transpose())


if __name__ == '__main__':
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=args.data_dir)

    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=args.intent_type, intent_file='{}/intents.json'.format(args.data_dir)) if args.intent_type else (None, None)

    # pp added: data loaders
    train_loader = multiwoz_dataloader.get_loader('{}/train_dials.json'.format(args.data_dir), input_lang_word2index, output_lang_word2index, args.intent_type, intent2index, batch_size=args.batch_size)
    # valid_loader_list = multiwoz_dataloader.get_loader_by_full_dialogue('{}/val_dials.json'.format(args.data_dir), input_lang_word2index, output_lang_word2index, args.intent_type, intent2index)
    # test_loader_list = multiwoz_dataloader.get_loader_by_full_dialogue('{}/test_dials.json'.format(args.data_dir), input_lang_word2index, output_lang_word2index, args.intent_type, intent2index)
    # Load validation file list:
    with open('{}/val_dials.json'.format(args.data_dir)) as outfile:
        val_dials = json.load(outfile)
    # Load test file list:
    with open('{}/test_dials.json'.format(args.data_dir)) as outfile:
        test_dials = json.load(outfile)

    delex_path = '%s/multi-woz/delex.json' % args.data_dir

    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index, intent2index, index2intent)
    # model = nn.DataParallel(model, device_ids=[0,1]) # latter for parallel
    model = model.to(detected_device)
    if args.load_param:
        model.loadModel(args.epoch_load)


    trainIters(model, intent2index, n_epochs=args.max_epochs, args=args)
