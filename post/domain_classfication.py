#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : Do the domain classification to show whether the characteristics of different domains are different
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-05-10
"""
from utils import util, multiwoz_dataloader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from plot_tsne import *

# Scale and visualize the embedding vectors
def plot_embedding(X, y, target, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def loadData(data_dir='../multiwoz-moe/data',intent_type='domain'):
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = util.loadDictionaries(mdir=data_dir)
    # pp added: load intents
    intent2index, index2intent = util.loadIntentDictionaries(intent_type=intent_type, intent_file='{}/intents.json'.format(data_dir)) if intent_type else (None, None)
    # pp added: data loaders
    batch_size = 8438 # 8438 # load all data here
    train_loader = multiwoz_dataloader.get_loader('{}/train_dials.json'.format(data_dir), input_lang_word2index, output_lang_word2index, intent_type, intent2index, batch_size=batch_size)
    data = iter(train_loader).next()
    input_tensor, _, target_tensor, _, bs_tensor, db_tensor, mask_tensor = data

    y10 = mask_tensor.transpose(0, 2).squeeze(0)
    y = [int(y10[i].min(0)[1]) for i in range(len(y10))] # [batch, 1], labels
    X = np.concatenate((input_tensor, target_tensor, bs_tensor, db_tensor), 1)  # [batch, feature]

    # plot_cluster(X, y, y)
    data_zs = pd.DataFrame(X)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    #
    # # plot_embedding(X, y, target=y, title='Origin')
    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9}, label=y)


    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

    return input_tensor


if __name__ == "__main__":
    loadData()