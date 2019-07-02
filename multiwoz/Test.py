from multiwoz.Evaluators import *

random.seed(1)

# diag={}
# for filename, dialogues in json.load(open('data/test_dials.json')).items():
#     diag[filename] = dialogues['sys']
# evaluateModel(diag, json.load(open('data/test_dials.json')), mode='test')

evaluator=MultiWozEvaluator('MultiWozEvaluator')

diag={}
# for filename, dialogues in evaluator.delex_dialogues.items():
#     one_diag=[]
#     for t, sent_t in enumerate(dialogues['log']):
#         if t%2==1:
#             one_diag.append(sent_t['text'])
#     diag[filename]=one_diag

# print(evaluator.evaluate_match_success(evaluator.delex_dialogues, mode='rollout'))
# random.seed(1)

# for filename, dialogues in json.load(open('data/test_dials.json')).items():
#     diag[filename] = dialogues['sys']
#
# print(evaluator.evaluate_match_success(diag))
# print(evaluator.evaluate_bleu_prf(diag))
path_bsl = 'results/bsl_20190510161309/data/test_dials/test_dials_gen.json'
path_moe = 'results/moe1_20190510165545/data/test_dials/test_dials_gen.json'
with open(path_moe) as fr:
    test_dials = json.load(fr)
    # print(evaluator.evaluate_match_success(test_dials))
    # print(evaluator.evaluate_bleu_prf(test_dials))
    evaluator.summarize_report(test_dials)