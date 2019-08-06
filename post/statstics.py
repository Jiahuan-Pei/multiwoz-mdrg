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
import seaborn as sns
import matplotlib as mpl

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

def loadData(data_dir='data/multi-woz',intent_type='domain', plot_flag=False, ob_ids=None):
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
        'UNK':        10,
        'Attraction': 100,
        'Booking':    100,
        'Hotel':      100,
        'Restaurant': 100,
        'Taxi':       50,
        'Train':      100,
        'general':    100
    }

    tmp = []
    distributions = []
    plot_ids = ob_ids if ob_ids is not None else range(len(index2intent.keys())) # show partly if you assign some of the ids
    for i in plot_ids:
    # for i in range(len(index2intent.keys())):
        intent = index2intent[i]
        labels.append(capwords(intent))
        text = intent_text[intent]
        target_dict_k = collections.Counter(text.split())
        specials = [word for word in target_dict_k if word not in output_lang_word2index] # oov
        rm_words = stopwords.words() + punctuations + specials  # we filter these words
        rm_words = [w for w in rm_words if w not in output_lang_word2index] # remain vocab words
        for word in rm_words:
            del target_dict_k[word]
        print(intent, ':::', target_dict_k)
        y = [target_dict_k[output_lang_index2word['%s' % ii]] for ii in x]

        # print('Expert_i_probs_list=', y)
        distributions.append(y)

        # smooth curves
        num_of_point = 400*100
        x_new = np.linspace(x.min(), x.max(), num_of_point)
        # 1.
        # y_smooth = spline(x, y, x_new)
        # 2.
        func = interp1d(x, y, kind='nearest') # cubic
        y_smooth = func(x_new)
        tmp.append(y)
        if plot_flag:
            l_k, = plt.plot(x_new, y_smooth, color=colormap[i + 2], label=intent)
            lines.append(l_k)
            # show top-n
            start_h = 30
            N = 10
            for j in np.arange(start_h, N + start_h):
                k_max, v_max = target_dict_k.most_common()[j]
                max_indx = output_lang_word2index[k_max]
                plt.annotate('[%s]' % (re.sub(r'\[|\]', '', k_max)),
                             xytext=(max_indx, v_max + (j+2)*shift[intent]),
                             xy=(max_indx, v_max),
                             color=colormap[i + 2], fontsize='xx-large',
                             arrowprops=dict(facecolor='black',
                                             arrowstyle="simple",
                                             connectionstyle="arc3,rad=-0.1"))
            # show lowest-n
            start_l = 30
            for j in np.arange(start_l, N + start_l):
                k_min, v_min = target_dict_k.most_common()[-j]
                min_indx = output_lang_word2index[k_min]
                plt.annotate('(%s)' % (re.sub(r'\[|\]', '', k_min)),
                             xytext=(min_indx, v_min + (j+2)*shift[intent]),
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
    plt.show()
    # plt.savefig('post/Statistic.png')

def loadDataDF(data_dir='data/multi-woz',intent_type='domain', plot_flag=False, remained_plot_intents=None):
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=data_dir)
    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=intent_type, intent_file='{}/intents.json'.format(data_dir)) if intent_type else (None, None)

    # read data
    with open('{}/train_dials.json'.format(data_dir)) as f:
        train_dials = json.load(f)
        # print train_dials

    intent_text = {}
    intent_list = [capwords(it) if it != it.upper() else it for it in intent2index.keys()]
    plot_intent_list = remained_plot_intents if remained_plot_intents is not None else intent_list
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

    labels = []
    output_vocab_len = 400
    x = np.arange(output_vocab_len)
    lines = []
    shift = { # max_x, max_y, min_x, min_y
        'UNK': (-20, -60, 10, -100),
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
    # plot_ids = ob_ids if ob_ids is not None else range(len(index2intent.keys())) # show partly if you assign some of the ids
    # for i in plot_ids:
    target_dict_all = collections.Counter(None)
    for i in range(len(index2intent.keys())):
        intent = index2intent[i]
        labels.append(capwords(intent))
        text = intent_text[intent]
        target_dict_k = collections.Counter(text.split()) # calculate TF dict for each intent
        target_dict_all += target_dict_k
        specials = [word for word in target_dict_k if word not in output_lang_word2index] # oov
        rm_words = stopwords.words() + punctuations + specials  # we filter these words
        rm_words = [w for w in rm_words if w not in output_lang_word2index]  # remain vocab words
        for word in rm_words:
            del target_dict_k[word]
        print(intent, ':::', target_dict_k)
        y = [target_dict_k[output_lang_index2word['%s' % ii]] for ii in x]
        # print('Expert_i_probs_list=', y)
        distributions.append(y)
    target_dict_all_sored = collections.OrderedDict(target_dict_all.most_common())
    print(target_dict_all_sored)
    # put data to DataFrame obj
    df = pd.DataFrame(distributions).T
    df.columns = intent_list
    # sort x-index by some value
    df['FreqAll'] = df.sum(axis=1) # add all freqs from type intents
    vocab_word_list = [output_lang_index2word['%s' % ii] for ii in x]
    df['Word'] = vocab_word_list # add one column "Word"
    # print(df.head(20))
    # df = df.sort_values(by=['FreqAll'], ascending=False) # sort index by one column; not necessary for density
    # print(df.head(20))

    # plot figures
    df = df[plot_intent_list]  # extract part of data for plot
    df = df.fillna(0)
    # df.plot() # before scale
    df = df.divide(df.sum(axis=1), axis=0)  # calculate the percentage
    # df.plot() # after scale
    # df.plot.area(stacked=True, title='Area(Stacked)')
    # df.plot.kde(bw_method=0.3, xlim=[-1000,1000], ylim=[-0.0002,0.008]) # default=0.3, the larger the smoother
    df.plot.kde(bw_method=0.3, figsize=(8,3)) # default=0.3, the larger the smoother
    # df.to_excel('post/plot_df.xlsx')
    # for it in plot_intent_list:
    #     sns.kdeplot(df[it], shade=True)

    # df[0:100].plot.kde(title='Density - [0, 100]')
    # df[100:200].plot.kde(title='Density - [100, 200]')
    # df[200:300].plot.kde(title='Density - [200, 300]')
    # df[300:400].plot.kde(title='Density - [300, 400]')
    # df[0:200].plot.kde(title='Density - [0, 200]')
    # df[100:300].plot.kde(title='Density - [100, 300]')
    # df[200:400].plot.kde(title='Density - [200, 400]')
    # df.plot.box()
    # sns.distplot(df['Hotel'], kde=True)
    # for it in plot_intent_list:
    #     sns.kdeplot(df[it])
    sns.set_palette("hls")
    # plt.figure(figsize=(20, 10))
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('x')
    plt.ylabel('Density', fontsize=18)
    # plt.ylabel('')
    plt.legend(loc=1, fontsize=18) # upper right
    plt.tight_layout()
    # plt.set_size_inches()
    plt.savefig('post/statistic_%s.png' % intent_type)
    plt.show()


if __name__ == "__main__":
    # for i in range(8):
    #     loadData(intent_type='domain', plot_flag=True, ob_ids=[i])
    # loadData(intent_type='domain_act') 

    remained_plot_intents_domain = [
        # 'UNK',
        'Attraction',
        # 'Booking', # tick1
        'Hotel', # tick2
        'Restaurant', # tick3
        'Taxi', # tick4
        # 'Train',
        'General' # tick5
    ]
    loadDataDF(intent_type='domain', plot_flag=True, remained_plot_intents=remained_plot_intents_domain)

    '''
    # loadDataDF(intent_type='domain', plot_flag=True, remained_plot_intents=None)
    remained_plot_intents_act = [
        'Welcome',
        'Book',
        'Inform',
        'Request',
        'Recommend',
    ]
    loadDataDF(intent_type='sysact', plot_flag=True, remained_plot_intents=remained_plot_intents_act)
    
    '''
