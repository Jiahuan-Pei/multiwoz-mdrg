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
from models.evaluator import evaluateModel, evaluateModelOnIntent
from models.model import Model

def evalulation_on_domain(model_fold_name='bsl'):
    # Generated testset
    # model_fold_name = 'bsl_g'
    test_dials_gen_dir = '%s/%s/data/test_dials' % (result_dir, model_fold_name)
    with open('{}/test_dials_gen.json'.format(test_dials_gen_dir)) as fr:
        test_dials_gen = json.load(fr)

    print('Test Overall:')
    evaluateModel(test_dials_gen, test_dials, delex_path, mode='Test')

    for intent in intent2index.keys():
        print('\nTest Intent:', intent)
        evaluateModelOnIntent(test_dials_gen, test_dials, delex_path, intent, mode='Test')


if __name__ == "__main__":
    intent_type = 'domain'
    result_dir = 'results'
    data_dir = '../multiwoz1-moe/data'
    delex_path = '%s/multi-woz/delex.json' % data_dir

    # Golden testset
    with open('{}/test_dials.json'.format(data_dir)) as outfile:
        test_dials = json.load(outfile)

    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(
        mdir=data_dir)

    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=intent_type,
                                                             intent_file='{}/intents.json'.format(
                                                                 data_dir)) if intent_type else (None, None)
    # 1. bsl, test on h0
    # evalulation_on_domain('bsl_20190510161309')
    # 2. bsl_m1, test on h1
    # evalulation_on_domain('bsl_m1_20190510161313')

    # 3. bsl_m2
    # evalulation_on_domain('bsl_m2_20190510161318')

    # 4. bsl_m3
    # evalulation_on_domain('bsl_m3_20190510165545')

    # print('-'*50)
    # 5. moe1
    evalulation_on_domain('moe1_20190510165545')