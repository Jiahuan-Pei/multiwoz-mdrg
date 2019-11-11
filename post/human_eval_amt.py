#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : Generate the js file for Amazon Mechnical Turk
(1) S2S:    89.50, ep19 (34970, 35041, 35043)
(2) LaRL:   88.75 (paper reported: 93.79)
(3) MoGNet: 99.43, ep18 (36281, 37610, 37697)
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-10-23
"""
import json
import codecs
import csv
from collections import OrderedDict
import re
from tqdm import tqdm
import numpy as np
from scipy import stats
from utils import util
util.init_seed(1) # must fix randomness when using evaluator
from multiwoz.Evaluators import *
from multiwoz.Create_delex_data import get_summary_bstate
evaluator=MultiWozEvaluator('MultiWozEvaluator')

golden_path = 'data/multi-woz/test_dials.json'
delex_path = 'data/multi-woz/delex.json'
raw_path = 'data/multi-woz/data.json'
bsl_path = 'post/case_study/bsl-34970/test_dials_gen_19.json'
# larl_raw_path = 'post/case_study/rl-2019-10-22-02-06-13/test_file.txt'
# larl_raw_path = 'post/case_study/rl-2019-10-22-19-32-55/test_file.txt'
larl_raw_path = 'post/case_study/rl-2019-10-23-10-33-07/test_file.txt'
larl_path = larl_raw_path.replace('txt', 'json')
mog_path = 'post/case_study/mog-36281/test_dials_gen_18.json'
good_path = 'post/case_study/human_eval_good_examples.json'
bad_path = good_path.replace('good', 'bad')
context_response_path = 'post/case_study/context_response_for_human_eval.txt'
amt_context_response_path = 'post/case_study/MDRG_input.csv'

with open(bsl_path, 'r') as bsl_fr, \
        open(larl_path, 'r') as larl_fr, \
        open(mog_path, 'r') as mog_fr, \
        open(golden_path, 'r') as gold_fr, \
        open(delex_path, 'r') as delex_fr, \
        open(raw_path, 'r') as raw_fr:
    bsl_json = json.load(bsl_fr)
    larl_json = json.load(larl_fr)
    mog_json = json.load(mog_fr)
    gold_json = json.load(gold_fr)
    delex_json = json.load(delex_fr)
    raw_json = json.load(raw_fr)

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

# convert the larl results to multiwoz results formation
def larl2woz_output(larl_raw_path=larl_raw_path, larl_path=larl_path):
    # here will also print the evaluation results of LaRL by our evaluator; note that they only have 999 dials for test
    d = {}
    dd = OrderedDict() # keep the printed order same with th gold json
    plain_texts = open(larl_raw_path, 'r').read().split('----------------------------------------')
    for text in plain_texts:
        try:
            m = re.match(r'^(.*json).* Pred: (.*)', text.strip().replace('\n', ' '))
            fname, pred = m.group(1), m.group(2)
            if fname not in d:
                d[fname] = [pred]
            else:
                d[fname].append(pred)
        except:
            pass
            # print(text)
    for k in gold_json:
        if k in d:
            dd[k] = d[k]
    with open(larl_path, 'w') as fw:
        json.dump(dd, fw, indent=4)
    return dd

# find good/bad dialogue examples
def choose_dialogues():
    theta = 0 # threshold
    topn = 200 # only choose the TopN best
    lown = 200 # only choose the LowN best
    countT = 0
    countL = 0

    bsl_score_list = []
    larl_score_list = []
    mog_score_list = []
    good_examples = OrderedDict()
    bad_examples = {}

    if os.path.exists(good_path) and False:
        print('Examples of dialogues exist, loading them...')
        with open(good_path, 'r') as fg, open(bad_path, 'r') as fb:
            sorted_good_examples = json.load(fg)
            sorted_bad_examples = json.load(fb)
    else:
        print('Selecting examples of dialogues...')
        for fname in tqdm(gold_json):
            # print('Processing==>', fname)
            if fname not in larl_json:
                print('LARL without testing file:', fname)
                # larl_json[fname] = bsl_json[fname]
                continue
            # if 'SNG' in fname:
            #     # print('PASS:', fname)
            #     continue
            bsl_score = evaluator.summarize_report({fname: bsl_json[fname]}, mode='Test', pt_values=False)[-1] # score = val[3]
            larl_score = evaluator.summarize_report({fname: larl_json[fname]}, mode='Test', pt_values=False)[-1] # score = val[3]
            mog_score = evaluator.summarize_report({fname: mog_json[fname]}, mode='Test', pt_values=False)[-1] # score = val[3]
            bsl_score_list.append(bsl_score)
            larl_score_list.append(larl_score)
            mog_score_list.append(mog_score)
            ml = mog_score - larl_score
            mb = mog_score - bsl_score
            if ml>0 and mb>0 and ml > theta and mb > theta:
                countT += 1
                good_examples[fname] = {
                    'ml_diff': ml,
                    'mb_diff': mb,
                    'user': gold_json[fname]['usr'],
                    'gold': gold_json[fname]['sys'],
                    'bsl': bsl_json[fname],
                    'larl': larl_json[fname],
                    'mog': mog_json[fname]
                }

            if ml<0 and mb<0 and -ml>theta and -mb>theta:
                countL += 1
                bad_examples[fname] = {
                    'lm_diff': -ml,
                    'bm_diff': -mb,
                    'user': gold_json[fname]['usr'],
                    'gold': gold_json[fname]['sys'],
                    'bsl': bsl_json[fname],
                    'larl': larl_json[fname],
                    'mog': mog_json[fname]
                }

        sorted_good_examples = OrderedDict(sorted(good_examples.items(), key=lambda t: t[1]['ml_diff'], reverse=True)[:topn])
        with open(good_path, 'w') as fg:
            json.dump(sorted_good_examples, fg, indent=4)

        sorted_bad_examples = OrderedDict(sorted(bad_examples.items(), key=lambda  t: t[1]['lm_diff'], reverse=True)[:lown])
        with open(bad_path, 'w') as fb:
            json.dump(sorted_bad_examples, fb, indent=4)

    return sorted_good_examples, sorted_bad_examples

# choose 1 context-response pair from each dialogue
def choose_context_response_pairs():
    fw = open(context_response_path, 'w')
    csvfile = codecs.open(amt_context_response_path, mode='w', encoding='utf-8')
    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['dial_name', 'turn_id', 'context', 'query', 'gold', 's2s', 'larl', 'mog', 'message'])
    dialogues, _ = choose_dialogues()
    count = 0
    print('&'*50, len(dialogues))
    for j, dial in enumerate(dialogues):
        n = len(dialogues[dial]['gold'])    # the number of responses in each dialogue
        diffs = []
        for i in range(n-1):
            utt_m = dialogues[dial]['mog'][i]
            utt_m_len = len(utt_m)
            utt_g = dialogues[dial]['gold'][i]
            utt_b = dialogues[dial]['bsl'][i]
            utt_l = dialogues[dial]['larl'][i]
            b = evaluator.summarize_report({dial: [utt_b]}, mode='Test', pt_values=False)[-1]
            l = evaluator.summarize_report({dial: [utt_l]}, mode='Test', pt_values=False)[-1]
            m = evaluator.summarize_report({dial: [utt_m]}, mode='Test', pt_values=False)[-1]
            # make sure the len will not affect the workers
            ld_ml = len(utt_m)/len(utt_l)
            ld_mb = len(utt_m)/len(utt_b)
            len_constraint = ld_ml>1.2 and ld_mb>1.1
            # if not len_constraint:
                # print(dial, i)
            val = (m - l) * (m - b) if len_constraint and m - l>0 and m - b>0 else 0
            diffs.append(val)
        if max(diffs) == 0:
            print('*'*5, dial)
            continue
        max_i = np.argmax(diffs)
        hist = []
        for k in range(int(max_i)):
            hist.extend(['U%s: %s'%(k+1, dialogues[dial]['user'][k]), 'S%s: %s'% (k+1, dialogues[dial]['gold'][k])])

        hist_str = '\n'.join(hist) if len(hist)>0 else 'nan'
        # write input file for AMT
        hist_html_str = '<br/>'.join(hist) if len(hist)>0 else 'nan'
        # ['dial_name', 'turn_id', 'context', 'query', 'gold', 's2s', 'larl', 'mog', 'message']
        count += 1
        if count> 100:
            break
        spamwriter.writerow([dial, max_i, hist_html_str, dialogues[dial]['user'][max_i],
                             dialogues[dial]['gold'][max_i], dialogues[dial]['bsl'][max_i],
                             dialogues[dial]['larl'][max_i], dialogues[dial]['mog'][max_i],
                             '<br/>'.join(raw_json[dial]['goal']['message'])])
        print('%s%s%s'%(count, '='*50, dial)), fw.write('%s%s%s\n'%(count, '='*50, dial))
        # print('MESSAGE:\n%s' % '\n'.join(raw_json[dial]['goal']['message'])), fw.write('MESSAGE:\n%s\n' % '\n'.join(raw_json[dial]['goal']['message']))
        print('DIALOGUE HISTORY:\n%s' % hist_str), fw.write('DIALOGUE HISTORY:\n%s\n' % hist_str)
        print('-'*50), fw.write('-'*50+'\n')
        print('USER: %s' % dialogues[dial]['user'][max_i]), fw.write('USER: %s\n' % dialogues[dial]['user'][max_i])
        print('GOLD: %s' % dialogues[dial]['gold'][max_i]), fw.write('GOLD: %s\n' % dialogues[dial]['gold'][max_i])
        print('S2S: %s' % dialogues[dial]['bsl'][max_i]), fw.write('S2S: %s\n' % dialogues[dial]['bsl'][max_i])
        print('LARL: %s' % dialogues[dial]['larl'][max_i]), fw.write('LARL: %s\n' % dialogues[dial]['larl'][max_i])
        print('MOG: %s' % dialogues[dial]['mog'][max_i]), fw.write('MOG: %s\n' % dialogues[dial]['mog'][max_i])
    fw.close()
    csvfile.close()
    return

if __name__ == "__main__":
    # do it for the first time: transform LaRL outputs to multiwoz formation
    if not os.path.exists(larl_path):
        larl2woz_output()
    choose_context_response_pairs()
