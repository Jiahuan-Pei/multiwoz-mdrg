#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-05-13
"""
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools as ft

def auto_display_subplots(df_list, title_list, column=2, label_names=None):
    label_names = df_list[0].columns.tolist() if label_names is None else label_names
    c = column  # num of columns
    n = len(df_list)
    r = n // c if n % c == 0 else n // c + 1
    fig, axes = plt.subplots(r, c, constrained_layout=True)
    handles = []
    for i in range(n):
        if i < n:
            row = i // c
            col = i % c
            # print(i, n, row, col)
            l = df_list[i].plot(
                # y=label_names,
                ax=axes[row, col],
                # title=title_list[i],
                sharex=True,
                sharey=True,
                legend=False,
                fontsize=8,
                xticks=range(0, len(df_list[i]) + 1, 5),
            )
            handles.append(l)
            axes[row, col].set_title(title_list[i], fontsize=8)
        else:
            # axes[1, 2].legend().set_visible(True)
            break

    fig.legend(handles,  # The line objects
               labels=label_names,  # The labels for each line
               loc="upper center",  # Position of legend
               bbox_to_anchor=(0.5, 0.06),
               # ncol=len(label_names),
               ncol=len(use_names),
               borderaxespad=0.0,  # Small spacing around legend box
               title="",  # Title for the legend
               fontsize=8,
               )
    return
exp_output_dir = 'results/train_ratio_1-9/'
statistics = {}
metric_list = ['Inform', 'Success', 'BLEU', 'Score']
use_names = ['Matches', 'Matches', 'BLEU', 'Score']
lines = []
model_list = []
df_list = []
# get all data & make data frame
for fname in os.listdir(exp_output_dir):
    model, step, jobid, _ = re.split(r'_|-|\.', fname)
    if model not in model_list:
        model_list.append(model)
    with open('{}{}'.format(exp_output_dir, fname)) as fr:
        l = []
        for line in fr.readlines()[-5:-1]:
            _, metric_name, val = line.split()
            item = metric_name, model, int(step), float(val)
            l.append(item)
        lines.extend(l)
d = pd.DataFrame(lines, columns=['metric', 'model', 'step', 'val'])

# auto_display_subplots(df_list, title_list, column=2, label_names=None)
for metric in metric_list:
    metric = 'Matches' if metric=='Inform' else metric
    d_metric = d[d.metric==metric]

    # make a dataframe for each models; cols[step, model1_val, model2_val, ...]
    frames_model = []
    for model in model_list:
        d_model = d_metric[d_metric.model == model].sort_values(by=['step'])[['step', 'val']]  # extract data for each models
        d_model = d_model.rename(columns={'val': model}) # rename val column

        frames_model.append(d_model)

    # merge dataframe all types of mokdels
    df = ft.reduce(lambda df1, df2: df1.merge(df2, "outer"), frames_model)
    df = df.set_index('step')  # reindex
    # df = df[model_list]
    df_list.append(df)
    print('-'*50, '\n', metric)
    print(df)

auto_display_subplots(df_list, metric_list, 2)
plt.savefig('{}{}'.format(exp_output_dir, 'comparision.png'))

if __name__ == "__main__":
    pass