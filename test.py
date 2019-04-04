#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from utils import util, multiwoz_dataloader
from model.evaluator import evaluateModel
from model.model import Model
from utils.util import detected_device

# pp added: print out env
util.get_env_info()

parser = argparse.ArgumentParser(description='multiwoz-bsl-te')
# 1. Data & Dir
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='data', help='the root directory of data')
# data_arg.add_argument('--model_path', type=str, default='results/bsl_g/model/translate.ckpt', help='Path to a specific model checkpoint.')
data_arg.add_argument('--model_dir', type=str, default='results/bsl_g/model/', help='Path to a specific model checkpoint')
# parser.add_argument('--original', type=str, default='results/bsl_g/model/', help='Original dir.')
data_arg.add_argument('--model_name', type=str, default='translate.ckpt')
data_arg.add_argument('--valid_output', type=str, default='results/bsl_g/data/val_dials/', help='Validation Decoding output dir path')
data_arg.add_argument('--decode_output', type=str, default='results/bsl_g/data/test_dials/', help='Decoding output dir path')

# 2. MISC
misc_arg = parser.add_argument_group('Misc')
misc_arg.add_argument('--dropout', type=float, default=0.0)
misc_arg.add_argument('--use_emb', type=str, default='False')
misc_arg.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
misc_arg.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
misc_arg.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
misc_arg.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')
# 3. Here add new args
new_arg = parser.add_argument_group('New')
new_arg.add_argument('--intent_type', type=str, default=None, help='separate experts by intents: None, domain, sysact or domain_act') # pp added

args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
print('args.device={}'.format(args.device))

# torch.manual_seed(args.seed)
util.init_seed(args.seed)

def load_config(args):
    config = util.unicode_to_utf8(
        # json.load(open('%s.json' % args.model_path, 'rb')))
        json.load(open('{}/{}.json'.format(args.model_dir, args.model_name), 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def loadModelAndData(num):
    # Load dictionaries
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=args.data_dir)
    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=args.intent_type, intent_file='{}/intents.json'.format(args.data_dir)) if args.intent_type else (None, None)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index, intent2index)
    model = model.to(detected_device)
    if args.load_param:
        model.loadModel(iter=num)

    # Load data
    if os.path.exists(args.decode_output):
        shutil.rmtree(args.decode_output)
        os.makedirs(args.decode_output)
    else:
        os.makedirs(args.decode_output)

    if os.path.exists(args.valid_output):
        shutil.rmtree(args.valid_output)
        os.makedirs(args.valid_output)
    else:
        os.makedirs(args.valid_output)

    # # Load validation file list:
    with open('{}/val_dials.json'.format(args.data_dir)) as outfile:
        val_dials = json.load(outfile)
    #
    # # Load test file list:
    with open('{}/test_dials.json'.format(args.data_dir)) as outfile:
        test_dials = json.load(outfile)

    return model, val_dials, test_dials, input_lang_word2index, output_lang_word2index, intent2index, index2intent


def decode(num=1):

    model, val_dials, test_dials, input_lang_word2index, output_lang_word2index, intent2index, index2intent  = loadModelAndData(num)

    delex_path = '%s/multi-woz/delex.json' % args.data_dir

    start_time = time.time()
    for ii in range(2):
        if ii == 0:
            continue  # added for debug; ignore greedy search part
            print(50 * '-' + 'GREEDY')
            model.beam_search = False
        else:
            print(50 * '-' + 'BEAM')
            model.beam_search = True

        # VALIDATION
        val_dials_gen = {}
        valid_loss = 0
        for name, val_file in val_dials.items():
            loader = multiwoz_dataloader.get_loader_by_dialogue(val_file, name,
                                                              input_lang_word2index, output_lang_word2index,
                                                              args.intent_type, intent2index)
            data = iter(loader).next()
            # Transfer to GPU
            if torch.cuda.is_available():
                data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
            input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor = data

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor, mask_tensor)

            valid_loss += loss_sentence
            val_dials_gen[name] = output_words

        print('Current VALID LOSS:', valid_loss)
        try:
            with open(args.valid_output + 'val_dials_gen.json', 'w') as outfile:
                json.dump(val_dials_gen, outfile, indent=4)
        except:
            print('json.dump.err.valid')

        evaluateModel(val_dials_gen, val_dials, delex_path, mode='valid')

        # TESTING
        test_dials_gen = {}
        test_loss = 0
        for name, test_file in test_dials.items():
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
            test_loss += loss_sentence
            test_dials_gen[name] = output_words

        test_loss /= len(test_dials)

        print('Current TEST LOSS:', test_loss)
        try:
            with open(args.decode_output + 'test_dials_gen.json', 'w') as outfile:
                json.dump(test_dials_gen, outfile, indent=4)
        except:
            print('json.dump.err.test')
        evaluateModel(test_dials_gen, test_dials, delex_path, mode='test')

    print('TIME:', time.time() - start_time)


def decodeWrapper():
    # Load config file
    # with open(args.model_path + '.config') as f:
    with open('{}/{}.config'.format(args.model_dir, args.model_name)) as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    # args.original = args.model_path
    for ii in range(1, args.no_models + 1):
        print(70 * '-' + 'EVALUATING EPOCH %s' % ii)
        # args.model_path = args.model_path + '-' + str(ii)
        with torch.no_grad():
            decode(ii)
        # try:
        #     decode(ii, intent2index)
        # except:
        #     print('cannot decode')

        # args.model_path = args.original

if __name__ == '__main__':
    decodeWrapper()
