#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : Case study & significance test on:
(1) BSL:   89.50, ep19 (34970, 35041, 35043);
(2) RMOE:  95.11, ep18 (36259, 36295, 37295, 37235, 37561)
(3) RPMOE: 99.43, ep18 (36281, )
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-08-05
"""
import json
import pandas as pd
from scipy import stats
from utils import util
util.init_seed(1) # must fix randomness when using evaluator
from multiwoz.Evaluators import *

evaluator=MultiWozEvaluator('MultiWozEvaluator')

golden_path = 'data/multi-woz/test_dials.json'
bsl_path = 'post/case_study/bsl-34970/test_dials_gen_19.json'
rmoe_path = 'post/case_study/rmoe-36259/test_dials_gen_18.json'
pmoe_path = 'post/case_study/pmoe-36281/test_dials_gen_18.json'

def ttest(data_A, data_B, alpha):
    # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b. for one sided test we multiply p-value by half
    t_results = stats.ttest_rel(data_A, data_B)
    # correct for one sided test
    pval = float(t_results[1]) / 2
    if (float(pval) <= float(alpha)):
        print("\nTest result is significant with p-value: {}".format(pval))
        return
    else:
        print("\nTest result is not significant with p-value: {}".format(pval))
        return

def load_cases(bsl=bsl_path, rmoe=rmoe_path, pmoe=pmoe_path, golden=golden_path):
    with open(bsl, 'r') as bsl_fr, \
         open(rmoe, 'r') as rmoe_fr, \
         open(pmoe, 'r') as pmoe_fr, \
         open(golden, 'r') as gold_fr:
        bsl_json = json.load(bsl_fr)
        rmoe_json = json.load(rmoe_fr)
        pmoe_json = json.load(pmoe_fr)
        gold_json = json.load(gold_fr)
        bsl_score_list = [] # samples that rmoe outperforms bsl
        rmoe_score_list = []
        pmoe_score_list = []
        good_examples = {}

        for fname in gold_json:
            bsl_score = evaluator.summarize_report({fname: bsl_json[fname]}, mode='Test', pt_values=False)[3] # score = val[3]
            rmoe_score = evaluator.summarize_report({fname: rmoe_json[fname]}, mode='Test', pt_values=False)[3] # score = val[3]
            pmoe_score = evaluator.summarize_report({fname: pmoe_json[fname]}, mode='Test', pt_values=False)[3] # score = val[3]
            bsl_score_list.append(bsl_score)
            rmoe_score_list.append(rmoe_score)
            pmoe_score_list.append(pmoe_score)
            if pmoe_score > rmoe_score and rmoe_score > bsl_score:
                good_examples[fname] = (pmoe_score-rmoe_score,
                                        rmoe_score-bsl_score,
                                        gold_json[fname]['usr'],
                                        gold_json[fname]['sys'],
                                        bsl_json[fname],
                                        rmoe_json[fname],
                                        pmoe_json[fname])
                # print('=' * 50)
                # print(fname)
                # print('USER:\n', gold_json[fname]['usr'])
                # print('SYS:\n', gold_json[fname]['sys'])
                # print('='*50)
                # print(bsl_json[fname], '\n'+'-'*50)
                # print(rmoe_json[fname], '\n'+'-'*50)
                # print(pmoe_json[fname], '\n'+'-'*50)

        ttest(bsl_score_list, rmoe_score_list, 0.01)
        ttest(bsl_score_list, pmoe_score_list, 0.01)
        ttest(rmoe_score_list, pmoe_score_list, 0.01)
        print('Find out good examples:', len(good_examples)) # 249
        df = pd.DataFrame(good_examples).T
        df.columns = ['pr_diff', 'rb_diff', 'user', 'gold', 'bsl', 'rmoe', 'pmoe']
        df = df.sort_values(by=['pr_diff','rb_diff'], ascending=False)
        print(df.head())
        df.to_excel('post/good_examples.xlsx')
    return

if __name__ == "__main__":
    load_cases()