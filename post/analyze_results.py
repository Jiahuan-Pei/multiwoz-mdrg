#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-04-11
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def auto_display_subplots(df_list, title_list, column=3, label_names=None):
    label_names = df_list[0].columns.tolist() if label_names is None else label_names
    c = column  # num of columns
    n = len(df_list)
    r = n // c if n % c == 0 else n // c + 1
    fig, axes = plt.subplots(r, c)
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
               ncol=len(use_names),
               borderaxespad=0.0,  # Small spacing around legend box
               title="",  # Title for the legend
               fontsize=8,
               )
    return

# one file
def old_one_jobout_to_df(job_out_file, fdir='post/', use_names=None, plot=False):
    with open('{0}{1}'.format(fdir, job_out_file), 'r') as fr:
        content = fr.read()
        # fix inconsistency -- START
        correct_dict = [
            (r'LOSS', r'Loss'),
            (r'(?<=\))\sLoss:', r'\nTrain Loss:'),
            (r'- Current ', ''),
            (r'BLEUS', r'BLEU'),
            (r'BLUES', r'BLEU'),
            (r'\s+:', r':'),
            (r'Valid Corpus', r'Valid'),
            (r'\bCorpus', r'Test'),
            (r'Valid Total number of dialogues:', r'Valid Dialogues:'),
            (r'\bTotal number of dialogues:', r'Test Dialogues:'),
            (r'Valid Test', r'Valid'),
            (r', Grad', r'\nTrain Grad'),
            (r', Loss act', r'\nTrain Loss Act')
        ]

        for k, v in correct_dict:
            content = re.sub(pattern=k, repl=v, string=content)
        # fix inconsistency -- END
        # print content
        use_names_default = [
            'Train Loss', 'Train Grad',
            'Valid Loss',
            'Valid BLEU', 'Valid Matches', 'Valid Success',
            # 'Valid Score',
            'Valid Dialogues',
            'Test BLEU', 'Test Matches', 'Test Success',
            # 'Test Score',
            'Test Dialogues',
        ]

        name_pattern_dict = {k: r'(?<=%s:\s)(\d+.{0,1}\d+)(?=\D)'%k for k in use_names_default}

        use_names = use_names if use_names is not None else use_names_default

        data = map(lambda x: re.findall(pattern=x, string=content), [name_pattern_dict[name] for name in use_names])
        df = pd.DataFrame(zip(*data), columns=use_names, dtype=np.float)
        # print(df)
        # scale BLEU score to get clear view
        for name in ['Valid BLEU', 'Test BLEU']:
            if name in df.columns.tolist():
                df.loc[:, name] *= 100
        for name in ['Valid Score' , 'Test Score']:
            mode = name.split()[0]
            col_name = df.columns.tolist()
            if '%s Matches' % mode in col_name and '%s Success' % mode in col_name and '%s BLEU' % mode in col_name:
                col_name.insert(col_name.index('%s Success' % mode)+1, name)  # add after it
                df.reindex(columns=col_name)
                df[name] = 0.5*df['%s Matches' % mode] + 0.5*df['%s Success' % mode] + 1*df['%s BLEU' % mode]
                print df
        if plot:
            df.plot()
            plt.show()
    return df

# multiple files
def old_mul_jobout_to_df(job_out_file_list, fdir='post/', outfile='output', use_names=None , plot=False, plot_split=None):
    new_names = []
    df_list = []
    jobid_list = []
    for f in job_out_file_list:
        jobid = re.findall(pattern=r'(?<=-)(\d+.{0,1}\d+)(?=.)', string=f)[0]
        jobid_list.append(jobid)
        df = old_one_jobout_to_df(f, fdir=fdir, use_names=use_names)
        df_list.append(df)
        new_names.extend(['%s_%s'%(name, jobid) for name in df.columns.tolist()])
    new_df = pd.concat(df_list, axis=1)
    new_df.columns = new_names
    extend_use_names = df_list[0].columns.tolist() # extend Score field
    writer = pd.ExcelWriter('%s%s_all_df.xlsx' % (fdir, outfile), engine='xlsxwriter')
    new_df.to_excel(excel_writer=writer, float_format = "%0.6f")
    writer.save()
    # new_df.columns = new_names
    if plot:
        # print(new_df)
        if plot_split is None:
            new_df.plot()
        elif plot_split == 'jobid':
            # new_df.plot(subplots=True, layout=(2, 5))
            auto_display_subplots(df_list, jobid_list, column=3)
        elif plot_split == 'names':
            df_group_by_names_list = []
            writer = pd.ExcelWriter('%s%s_split_df.xlsx' % (fdir, outfile), engine='xlsxwriter')
            for name in extend_use_names:
                col_names = [new_name for new_name in new_names if name in new_name]
                df_group_by_names = new_df[col_names]
                df_group_by_names_list.append(df_group_by_names)
                df_group_by_names.to_excel(excel_writer=writer, float_format="%0.6f", sheet_name=name)
            writer.save()
            auto_display_subplots(df_group_by_names_list, extend_use_names, column=3, label_names=jobid_list)
        plt.suptitle(outfile, fontsize=12)
        plt.tight_layout()
        plt.show()
    return new_df

# ====== Different settings of experiments --START =======

# ====== Different settings of experiments --END =======

if __name__ == "__main__":
    job_out_file_list = [
        'bsl_g_0-295169.out',
        'bsl_g_0-295170.out',
        'bsl_g_0-295256.out',
        'bsl_g_0-295257.out',
        'bsl_g_0-295258.out'
    ]
    use_names = [
        'Train Loss',
        'Valid Loss',
        'Valid BLEU',
        'Valid Matches',
        'Valid Success',
        'Test BLEU',
        'Test Matches',
        'Test Success',
    ]
    # old_mul_jobout_to_df(job_out_file_list, fdir='post/', outfile='bsl64valid', use_names=use_names, plot=True, plot_split='jobid')
    old_mul_jobout_to_df(job_out_file_list, fdir='post/', outfile='bsl64valid', use_names=use_names, plot=True, plot_split='names')
