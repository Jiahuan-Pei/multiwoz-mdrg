#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-06-03
"""
from utils import util, multiwoz_dataloader
from utils.util import PAD_token, SOS_token, EOS_token, UNK_token
import collections
import re
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
# plt.style.use('ggplot')

# import matplotlib as mpl
# colormap = mpl.cm.Dark2.colors
import brewer2mpl
bmap = brewer2mpl.get_map('Paired', 'qualitative', 12)
colormap = bmap.mpl_colors
import seaborn as sns

from scipy.interpolate import spline, interp1d
from scipy.stats import entropy
from itertools import combinations

from nltk.corpus import stopwords
stopwords.words('english')
from string import punctuation, capwords
punctuations = list(punctuation)

# Kullback-Leibler divergence
def KL(p, q):
    return entropy(p, q)

# Average KL
def AKL(expert_prob_list):
    epsilon = 10e-2
    KL_list = []
    for pi, pj in combinations([[np.float(e)+epsilon for e in l] for l in expert_prob_list], 2): # c_n^2
        # KL_list.append(np.square(entropy(pi, pj)))
        KL_list.append(entropy(pi, pj))
    return np.mean(KL_list)

def loadData(data_dir='data/multi-woz',intent_type='domain', plot_flag=False):
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=data_dir)
    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=intent_type, intent_file='{}/intents.json'.format(data_dir)) if intent_type else (None, None)

    # read data
    with open('{}/train_dials.json'.format(data_dir)) as f:
        train_dials = json.load(f)
        # print train_dials

    intent_text = {}
    for name, dial in train_dials.items(): # dial
        for sys, acts in zip(dial['sys'], dial['acts']): # turn
            for intent in intent2index.keys():
                if intent in ' '.join(acts):
                    if intent not in intent_text:
                        intent_text[intent] = sys
                    else:
                        intent_text[intent] += ' ' + sys
                else:
                    pass
                    # if intent == 'Hotel-Request':
                    #     print('00000')
                    #     print('NULL')

    labels = []
    output_vocab_len = 400
    x = np.arange(output_vocab_len)
    lines = []
    shift = { # max_x, max_y, min_x, min_y
        'UNK': (-20, -60, 10, -50),
        'Attraction': (15, 10, -30, 160),
        'Booking': (-60, 10, 10, 200),
        'Hotel': (50, 10, 10, -50),
        'Restaurant': (15, 10, 20, 100),
        'Taxi': (-50, 10, 30, -50),
        'Train': (30, 10, 0, 150),
        'general': (-40, 10, 20, -50)
    }

    tmp = []
    distributions = []
    for i in range(len(index2intent.keys())):
        intent = index2intent[i]
        labels.append(capwords(intent))
        text = intent_text[intent]
        target_dict_k = collections.Counter(text.split())
        specials = [word for word in target_dict_k if word not in output_lang_word2index] # oov
        rm_words = stopwords.words() + punctuations + specials  # we filter these words
        for word in rm_words:
            del target_dict_k[word]
        print(intent, ':::', target_dict_k)
        y = [target_dict_k[output_lang_index2word['%s' % ii]] for ii in x]

        # print('Expert_i_probs_list=', y)
        distributions.append(y)

        # smooth curves
        xnew = np.linspace(x.min(), x.max(), 400)
        # 1.
        y = spline(x, y, xnew)
        # 2.
        # func = interp1d(x, y, kind='nearest') # cubic
        # y = func(xnew)
        tmp.append(y)
        if plot_flag:
            l_k, = plt.plot(x, y, color=colormap[i + 2], label=intent)
            lines.append(l_k)
            # show top-n
            N = 1
            for j in np.arange(N):
                k_max, v_max = target_dict_k.most_common()[j]
                max_indx = output_lang_word2index[k_max]
                plt.annotate('[%s]' % (re.sub(r'\[|\]', '', k_max)),
                             # xytext=(max_indx + shift[intent][0], v_max + shift[intent][1]),
                             xy=(max_indx, v_max),
                             color=colormap[i + 2], fontsize='xx-large',
                             arrowprops=dict(facecolor='black',
                                             arrowstyle="simple",
                                             connectionstyle="arc3,rad=-0.1"))
            # show lowest-n
            start = 10
            for j in np.arange(start, N + start):
                k_min, v_min = target_dict_k.most_common()[-j]
                min_indx = output_lang_word2index[k_min]
                plt.annotate('(%s)' % (re.sub(r'\[|\]', '', k_min)),
                             # xytext=(min_indx + +shift[intent][2], v_min + shift[intent][3]),
                             xy=(min_indx, v_min),
                             color=colormap[i + 2], fontsize='xx-large',
                             arrowprops=dict(facecolor='blue',
                                             arrowstyle="simple",
                                             connectionstyle="arc3,rad=-0.1"), )

    # print(distributions)
    print(intent_type, '=', AKL(distributions))

    plt.legend(labels=labels)
    plt.xlabel('Index of a word')
    plt.ylabel('Frequency')
    # plt.grid()
    # plt.show()
    plt.savefig('post/Statistic.png')

if __name__ == "__main__":
    loadData(intent_type='domain', plot_flag=True)
    loadData(intent_type='sysact')
    loadData(intent_type='domain_act')
