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

for filename, dialogues in json.load(open('data/multi-woz/test_dials.json')).items():
    diag[filename] = dialogues['sys']
evaluator.summarize_report(diag)

path_bsl = 'results/test_dials_gen(bsl_m2_20190510161318).json'
path_moe = 'results/test_dials_gen(moe1_20190510165545).json'
with open(path_bsl) as fr:
    print(path_bsl)
    evaluator.summarize_report(json.load(fr))

with open(path_moe) as fr:
    print(path_moe)
    evaluator.summarize_report(json.load(fr))