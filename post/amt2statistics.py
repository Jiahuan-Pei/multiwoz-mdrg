#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-11-04
"""
import pandas as pd
from prettytable import PrettyTable

# path_v1 = 'post/Batch_3823227_batch_results_v1.csv'
path_v1 = 'post/Batch_3823227_batch_results_v1_293.csv'
# path_v2 = 'post/Batch_3823257_batch_results_v2_190.csv'
# path_v2 = 'post/Batch_3823257_batch_results_v2_250.csv'
# path_v2 = 'post/Batch_3823257_batch_results_v2_270.csv'
path_v2 = 'post/Batch_3823257_batch_results_v2_294.csv'


def count_data_v1():
    print('V1: Given golden response....')
    t = PrettyTable()
    df = pd.DataFrame.from_csv('post/Batch_3823227_batch_results_v1.csv', encoding='utf-8')
    n = len(df)
    models = ['S2S', 'LARL', 'MOG']
    criteria = ['Informativeness', 'Consistency', 'Fluency', 'Humanness']
    t.field_names = ['Model'] + criteria
    for model in models:
        row_list = [model]
        for c in criteria:
            val = (df['Answer.%s.%s'%(c, model)]==True).sum()
            print('(%s, %s, %s/%s)' % (model, c, val, n))
            val = val/n*100
            row_list.append('%.2f'%val)
        t.add_row(row_list)
    print(t)


def data_disambiguate_v1(theta = 1):
    print('data_disambiguate_v1: Given golden response....')
    t = PrettyTable()
    df = pd.DataFrame.from_csv(path_v1, encoding='utf-8')
    # df.groupby(['Input.dial_name', 'Input.turn_id'])
    new_df_list = []
    models = ['S2S', 'LARL', 'MOG']
    criteria = ['Informativeness', 'Consistency', 'Fluency', 'Humanness']
    t.field_names = ['Model'] + criteria
    for (name, turn), group in df.groupby(['Input.dial_name', 'Input.turn_id']):
        nworkers = len(group)
        # if the work do it less than 20% of the average time (or less than 10s)
        # group = group[group['WorkTimeInSeconds'] >= max(0.2 * group['WorkTimeInSeconds'].mean(), 10)]
        new_group = group.iloc[0]
        for model in models:
            for c in criteria:
                numTrue = (group['Answer.%s.%s'%(c, model)]==True).sum()
                # merge strategy for different workers with different labels
                if numTrue >= theta:
                    new_group['Answer.%s.%s' % (c, model)] = True
        new_df_list.append(new_group)
    new_df = pd.concat(new_df_list, axis=1, ignore_index=True).T

    n = len(new_df)
    print(n)
    for model in models:
        row_list = [model]
        for c in criteria:
            val = (new_df['Answer.%s.%s'%(c, model)]==True).sum()/n * 100
            row_list.append('%.2f'%val)
        t.add_row(row_list)
    print(t)


def count_data_v2(norm=False):
    print('V2: Put golden response as an option....Normalized by ground truth=%s'%norm)
    t = PrettyTable()
    df = pd.DataFrame.from_csv(path_v2, encoding='utf-8')
    models = ['S2S', 'LARL', 'MOG', 'GOLD']
    critera = ['Informativeness', 'Consistency']
    # critera = ['Informativeness', 'Consistency', 'Fluency', 'Humanness']
    t.field_names = ['Model'] + critera + ['Satisfactory']
    for model in models:
        row_list = [model]
        df['Answer.Satisfactory.%s' % model] = df['Answer.Informativeness.%s' % model] & df['Answer.Consistency.%s' % model]
        for c in critera + ['Satisfactory']:
            n = (df['Answer.%s.GOLD'%c]==True).sum() if norm else len(df)
            val = (df['Answer.%s.%s'%(c, model)]&df['Answer.%s.GOLD'%c]==True).sum() if norm else (df['Answer.%s.%s' % (c, model)]==True).sum()
            print('(%s, %s, %s/%s)' % (model, c, val, n))
            val = val/n*100
            # if only believe the one also choose the golden
            row_list.append('%.2f'%val)
        t.add_row(row_list)

    print(t)


def data_disambiguate_v2(norm=False, theta=1):
    print('data_disambiguate_v1: Given golden response....')
    t = PrettyTable()
    df = pd.DataFrame.from_csv(path_v2, encoding='utf-8')
    # df.groupby(['Input.dial_name', 'Input.turn_id'])
    new_df_list = []
    models = ['S2S', 'LARL', 'MOG', 'GOLD']
    criteria = ['Informativeness', 'Consistency']
    # criteria = ['Informativeness', 'Consistency', 'Fluency', 'Humanness']
    # criteria2 = ['Informativeness', 'Consistency', 'Satisfactory']

    t.field_names = ['Model'] + criteria + ['Satisfactory']

    for model in models:
        df['Answer.Satisfactory.%s' % model] = df['Answer.Informativeness.%s' % model] & df['Answer.Consistency.%s' % model]

    for (name, turn), group in df.groupby(['Input.dial_name', 'Input.turn_id']):
        nworkers = len(group)
        # if the work do it less than 20% of the average time (or less than 10s)
        # group = group[group['WorkTimeInSeconds'] >= max(0.2 * group['WorkTimeInSeconds'].mean(), 10)]
        new_group = group.iloc[0]
        for model in models:
            for c in criteria + ['Satisfactory']:
                numTrue = (group['Answer.%s.%s'%(c, model)]==True).sum()
                # merge strategy for different workers with different labels
                if numTrue >= theta:
                    new_group['Answer.%s.%s' % (c, model)] = True
        new_df_list.append(new_group)
    new_df = pd.concat(new_df_list, axis=1, ignore_index=True).T
    # add new column


    for model in models:
        row_list = [model]
        for c in criteria + ['Satisfactory']:
            flag = True # (model!='MOG')
            n = (new_df['Answer.%s.GOLD' % c]==True).sum() if norm and flag else len(new_df)
            val = (new_df['Answer.%s.%s'%(c, model)]&new_df['Answer.%s.GOLD'%c]==True).sum() if norm and flag else (new_df['Answer.%s.%s' % (c, model)]==True).sum()
            print('(%s, %s, %s/%s)' % (model, c, val, n))
            val = val/n * 100
            row_list.append('%.2f'%val)
        t.add_row(row_list)
    print(t)

if __name__ == "__main__":

    # 1. Given the golden label
    # count_data_v1()
    # data_disambiguate_v1(1)
    # data_disambiguate_v1(2)
    # data_disambiguate_v1(3)

    # 2. use Golden Test
    # default: norm = False, theta = 1
    # data_disambiguate_v2(False, 1)
    # data_disambiguate_v2(False, 2)
    # data_disambiguate_v2(False, 3)
    #
    data_disambiguate_v2(True,1)
    data_disambiguate_v2(True,2)
    data_disambiguate_v2(True,3)
    #
    # count_data_v2()
    # count_data_v2(True)
