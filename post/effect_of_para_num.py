#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : Compare the performance with similar size of model parameters
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-08-06
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# See the effect of the number of the parameters in RPMOE and S2SAttnGRU
def paras_effect():
    plt.figure(figsize=(8, 4))
    # only remain the best results after ep15?
    bsl = [
        (  574403, 89.5001), # bsl-150; 34970
        ( 1569653, 94.1281), # bsl-300; 37847,
        ( 2638553, 95.1219), # bsl-400; 38040, ep=16.
        ( 3987453, 89.3915), # bsl-500; 37844, ep=17.
        ( 5616353, 90.9570), # bsl-600; 38041, ep=12.
        ( 7525253, 89.4626), # bsl-700; 38043, ep=19.
        ( 9714153, 95.1428), # bsl-800; 38044, ep=8(95.1428).
        ( 10913603, 94.2035), # bsl-850; 38725, ep=.
        (12183053, 89.0027), # bsl-900; 38045, ep=15.
        (14931953, 91.1405), # bsl-1000; 37846; ep=19.
    ]


    pmoe = [
        # (  585699, 70.4917), # rpmoe-30 ; 37852; ep15
        (  962609, 78.0583), # rpmoe-40 ; 38192; ep13
        # ( 1442919, 80.6915), # rpmoe-50 ; 37850; ep15
        ( 3096069, 81.1689), # rpmoe-75 ; 38726; ep18
        # ( 3504249, 87.5060), # rpmoe-80 ; 38166；ep20
        ( 5395469, 86.5969), # rpmoe-100; 37849; ep19
        # ( 7700289, 89.5314), # rpmoe-120; 38167; ep16
        ( 9007799, 96.1105), # rpmoe-130; 38170; ep7
        # (10418709, 86.5386), # rpmoe-140; 38171; ep20
        (11933019, 99.4251), # rpmoe-150; 36281
        # (13550729, 88.0699), # rpmoe-160; 38711; ep10
        (15271839, 91.0914), # rpmoe-170; 38712; ep18

        # (12470061, 57.3818), # RL setting 38169; ep7
        # (21055569, 91.2603), # rpmoe-200; 38165; ep7
    ]

    bsl_df = pd.DataFrame(bsl, columns=["num", "score"]).set_index('num', drop=False)
    bsl_df['score'].plot(style='X-')
    pmoe_df = pd.DataFrame(pmoe, columns=["num", "score"]).set_index('num', drop=False)
    pmoe_df['score'].plot(style='o-')
    print(bsl_df.head())
    print(pmoe_df.head())
    plt.xlabel('Number of parameters')
    plt.ylabel('Score')
    plt.legend(['S2SAttnGRU', 'MoGNet'], loc=4)
    plt.tight_layout()
    # plt.show()
    plt.savefig('post/score_paranum.png')

def lambda_expert_effect():
    plt.figure(figsize=(8, 4))
    lambda_expert_scores = [
        (0.0, 92.4779), # 38093
        (0.1, 97.0249), # 38094
        # (0.2, 96.0355), # 38271
        (0.3, 96.5535), # 38273
        # (0.4, 98.6573), # 38276
        (0.5, 99.4251), # 36281
        # (0.6, 95.0848), # 38275
        (0.7, 99.9893), # 38274
        # (0.8, 99.083), # 38272
        (0.9, 92.6091), # 38095
        (0.95, 82.9739), # 38732
        # (0.9999, 12.3133), # 38727
        # (1.0, 12.3000) # 38277
    ]
    lambda_expert_df = pd.DataFrame(lambda_expert_scores, columns=["lambda", "score"]).set_index('lambda', drop=False)
    lambda_expert_df['score'].plot(style='X-')
    plt.xlabel('Lambda')
    plt.ylabel('Score')
    plt.legend(['MoGNet'], loc=4)
    plt.tight_layout()
    # plt.show()
    plt.savefig('post/score_lambda.png')

if __name__ == "__main__":

    paras_effect()
    lambda_expert_effect()