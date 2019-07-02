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
from models.evaluator import *
from models.model import Model
from utils.util import detected_device, pp_mkdir

# pp added: print out env
util.get_env_info()

parser = argparse.ArgumentParser(description='multiwoz1-bsl-te')
# 1. Data & Dir
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='data', help='the root directory of data')
# data_arg.add_argument('--model_path', type=str, default='results/bsl/models/translate.ckpt', help='Path to a specific models checkpoint.')
data_arg.add_argument('--model_dir', type=str, default='results/bsl/models/', help='Path to a specific models checkpoint')
# parser.add_argument('--original', type=str, default='results/bsl/models/', help='Original dir.')
data_arg.add_argument('--model_name', type=str, default='translate.ckpt')
data_arg.add_argument('--valid_output', type=str, default='results/bsl/data/val_dials/', help='Validation Decoding output dir path')
data_arg.add_argument('--decode_output', type=str, default='results/bsl/data/test_dials/', help='Decoding output dir path')

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
new_arg.add_argument('--lambda_expert', type=float, default=0.5) # use xx percent of training data
new_arg.add_argument('--mu_expert', type=float, default=0.5) # use xx percent of training data
new_arg.add_argument('--gamma_expert', type=float, default=0.5) # use xx percent of training data
new_arg.add_argument('--debug', type=util.str2bool, nargs='?', const=True, default=False, help='if True use small data for debugging')
args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
print('args.device={}'.format(args.device))

# torch.manual_seed(args.seed)
util.init_seed(args.seed)
print(args)

def load_config(args):
    config = util.unicode_to_utf8(
        # json.load(open('%s.json' % args.model_path, 'rb')))
        json.load(open('{}{}.json'.format(args.model_dir, args.model_name), 'rb')))
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

    # # Load validation file list:
    with open('{}/val_dials.json'.format(args.data_dir)) as outfile:
        val_dials = json.load(outfile)
    #
    # # Load test file list:
    with open('{}/test_dials.json'.format(args.data_dir)) as outfile:
        test_dials = json.load(outfile)

    return model, val_dials, test_dials, input_lang_word2index, output_lang_word2index, intent2index, index2intent


def decode(num=1, beam_search=False):

    model, val_dials, test_dials, input_lang_word2index, output_lang_word2index, intent2index, index2intent  = loadModelAndData(num)

    delex_path = '%s/multi-woz/delex.json' % args.data_dir

    start_time = time.time()
    model.beam_search = beam_search

    step = 0 if not args.debug else 2 # small sample for debug

    # VALIDATION
    val_dials_gen = {}
    valid_loss = 0
    for name, val_file in list(val_dials.items())[-step:]:
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

    Valid_Score = evaluateModel(val_dials_gen, val_dials, delex_path, mode='valid')
    # evaluteNLG(val_dials_gen, val_dials)

    # TESTING
    test_dials_gen = {}
    test_loss = 0
    for name, test_file in list(test_dials.items())[-step:]:
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

    Test_Score = evaluateModel(test_dials_gen, test_dials, delex_path, mode='test')
    # evaluteNLG(test_dials_gen, test_dials)

    print('TIME:', time.time() - start_time)
    return Valid_Score, val_dials_gen, Test_Score, test_dials_gen


def decodeWrapper(beam_search=False):
    # Load config file
    # with open(args.model_path + '.config') as f:
    with open('{}{}.config'.format(args.model_dir, args.model_name)) as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            if k=='data_dir': # ignore this arg
                continue
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    # args.original = args.model_path
    Best_Valid_Score = None
    Best_Test_Score = None
    Best_model_id = 0
    Best_val_dials_gen = {}
    Best_test_dials_gen = {}
    for ii in range(1, args.no_models + 1):
        print(30 * '-' + 'EVALUATING EPOCH %s' % ii)
        # args.model_path = args.model_path + '-' + str(ii)
        with torch.no_grad():
            Valid_Score, val_dials_gen, Test_Score, test_dials_gen = decode(ii, beam_search)
            if Best_Valid_Score is None or Best_Valid_Score[-2] < Valid_Score[-2]:
                Best_Valid_Score = Valid_Score
                Best_Test_Score = Test_Score
                Best_val_dials_gen = val_dials_gen
                Best_test_dials_gen = test_dials_gen
                Best_model_id = ii
        # try:
        #     decode(ii, intent2index)
        # except:
        #     print('cannot decode')

    # save best generated output to json
    print('Summary'+'~'*50)
    print('Best model: %s'%(Best_model_id))
    BLEU, MATCHES, SUCCESS, SCORE, total = Best_Test_Score
    mode = 'Test'
    print('%s BLEU: %.4f' % (mode, BLEU))
    print('%s Matches: %2.2f%%' % (mode, MATCHES))
    print('%s Success: %2.2f%%' % (mode, SUCCESS))
    print('%s Score: %.4f' % (mode, SCORE))
    print('%s Dialogues: %s' % (mode, total))
    suffix = 'bm' if beam_search else 'gd'
    try:
        with open(args.valid_output + 'val_dials_gen_%s.json' % suffix, 'w') as outfile:
            json.dump(Best_val_dials_gen, outfile, indent=4)
    except:
        print('json.dump.err.valid')
    try:
        with open(args.decode_output + 'test_dials_gen_%s.json' % suffix, 'w') as outfile:
            json.dump(Best_test_dials_gen, outfile, indent=4)
    except:
        print('json.dump.err.test')

if __name__ == '__main__':
    # create dir for generated outputs of valid and test set
    pp_mkdir(args.valid_output)
    pp_mkdir(args.decode_output)
    print('\n\nGreedy Search'+'='*50)
    decodeWrapper(beam_search=False)
    print('\n\nBeam Search' + '=' * 50)
    decodeWrapper(beam_search=True)
    # evaluteNLGFile(gen_dials_fpath='results/bsl_20190510161309/data/test_dials/test_dials_gen.json',
    #                 ref_dialogues_fpath='data/test_dials.json')
    # evaluteNLGFiles(gen_dials_fpaths=['results/bsl_20190510161309/data/test_dials/test_dials_gen.json',
    #                                  'results/moe1_20190510165545/data/test_dials/test_dials_gen.json'],
                    # ref_dialogues_fpath='data/test_dials.json')
    # from nlgeval import compute_metrics
    # metrics_dict = compute_metrics(hypothesis='/Users/pp/Code/nlg-eval/examples/hyp.txt',
    #                                references=['/Users/pp/Code/nlg-eval/examples/ref1.txt'])

