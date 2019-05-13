#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-05-13
"""
from __future__ import division, print_function, unicode_literals

import json
from io import open

import torch

from utils import util, multiwoz_dataloader
from model.evaluator import evaluateModel, evaluateModelOnIntent
from model.model import Model

intent_type = 'domain'
model_fold_name = 'bsl_g'
result_dir = 'results'

data_dir='../multiwoz-moe/data'
delex_path = '%s/multi-woz/delex.json' % data_dir

# Golden testset
with open('{}/test_dials.json'.format(data_dir)) as outfile:
    test_dials = json.load(outfile)

# Generated testset
test_dials_gen_dir = '%s/%s/data/test_dials' % (result_dir, model_fold_name)
with open('{}/test_dials_gen.json'.format(test_dials_gen_dir)) as fr:
    test_dials_gen = json.load(fr)

input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(
    mdir=data_dir)

# pp added: load intents
intent2index, index2intent = util.loadIntentDictionaries(intent_type=intent_type,
                                                         intent_file='{}/intents.json'.format(
                                                             data_dir)) if intent_type else (None, None)

def evalulation_on_domain():
    print('Test Overall:')
    evaluateModel(test_dials_gen, test_dials, delex_path, mode='Test')

    for intent in intent2index.keys():
        print('\nTest Intent:', intent)
        evaluateModelOnIntent(test_dials_gen, test_dials, delex_path, intent, mode='Test')


if __name__ == "__main__":
    evalulation_on_domain()