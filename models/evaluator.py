import random
import sys
sys.path.append('..')

from utils.dbPointer import queryResultVenues
from utils.delexicalize import *
from utils.nlp import *

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']


def parseGoal(goal, d, domain):
    """Parses user goal into dictionary format."""
    goal[domain] = {}
    goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
    if 'info' in d['goal'][domain]:
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append('reference')
            if 'reqt' in d['goal'][domain]:
                if 'trainID' in d['goal'][domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if 'reqt' in d['goal'][domain]:
                for s in d['goal'][domain]['reqt']:  # addtional requests:
                    if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if 'book' in d['goal'][domain]:
                goal[domain]['requestable'].append("reference")

        goal[domain]["informable"] = d['goal'][domain]['info']
        if 'book' in d['goal'][domain]:
            goal[domain]["booking"] = d['goal'][domain]['book']

    return goal


def evaluateModel(dialogues, val_dials, delex_path, mode='Valid'):
    """Gathers statistics for the whole sets."""
    try:
        fin1 = open(delex_path, 'r')
    except:
        print('cannot find the delex file!=', delex_path)
    delex_dialogues = json.load(fin1)
    successes, matches = 0, 0
    total = 0

    gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}
    sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    for filename, dial in dialogues.items():
        data = delex_dialogues[filename]

        goal, _, _, requestables, _ = evaluateRealDialogue(data, filename)

        success, match, stats = evaluateGeneratedDialogue(dial, goal, data, requestables)

        successes += success
        matches += match
        total += 1

        for domain in gen_stats.keys():
            gen_stats[domain][0] += stats[domain][0]
            gen_stats[domain][1] += stats[domain][1]
            gen_stats[domain][2] += stats[domain][2]

        if 'SNG' in filename:
            for domain in gen_stats.keys():
                sng_gen_stats[domain][0] += stats[domain][0]
                sng_gen_stats[domain][1] += stats[domain][1]
                sng_gen_stats[domain][2] += stats[domain][2]

    # BLUE SCORE
    corpus = []
    model_corpus = []
    bscorer = BLEUScorer()

    count_wrong_len = 0
    for dialogue in dialogues:
        data = val_dials[dialogue]
        model_turns, corpus_turns = [], []
        for idx, turn in enumerate(data['sys']):
            corpus_turns.append([turn])
        for turn in dialogues[dialogue]:
            model_turns.append([turn])

        if len(model_turns) == len(corpus_turns):
            corpus.extend(corpus_turns)
            model_corpus.extend(model_turns)
        else:
            count_wrong_len += 1
            print('wrong length!!!')
            # print(model_turns)
    if count_wrong_len:
        print('count_wrong_len_ratio={}/{}'.format(count_wrong_len, len(dialogues)))
    # Print results
    try:
        BLEU = bscorer.score(model_corpus, corpus)
        MATCHES = (matches / float(total) * 100)
        SUCCESS = (successes / float(total) * 100)
        SCORE = 0.5 * MATCHES + 0.5 * SUCCESS + 100 * BLEU
        print('%s BLEU: %.4f' % (mode, BLEU))
        print('%s Matches: %2.2f%%' % (mode, MATCHES))
        print('%s Success: %2.2f%%' % (mode, SUCCESS))
        print('%s Score: %.4f' % (mode, SCORE))
        print('%s Dialogues: %s' % (mode, total))
        return BLEU, MATCHES, SUCCESS, SCORE, total
    except:
        print('SCORE ERROR')

def evaluateModelOnIntent(dialogues, val_dials, delex_path, intent, mode='Valid'):
    """Gathers statistics for the whole sets."""
    try:
        fin1 = open(delex_path, 'r')
    except:
        print('cannot find the delex file!=', delex_path)
    delex_dialogues = json.load(fin1)
    successes, matches = 0, 0
    total = 0
    total_turns = 0
    total_dials = 0

    gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}
    sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    for filename, dial in dialogues.items():
        data = delex_dialogues[filename]

        goal, _, _, requestables, _ = evaluateRealDialogue(data, filename)

        # filter goal & requestbles using domain
        new_goal = {}; new_req = {}
        for g in goal:
            if intent.lower() in g:
                new_goal[g] = goal[g]
        for r in requestables:
            if intent.lower() in r:
                new_req[r]=requestables[r]

        success, match, stats = evaluateGeneratedDialogue(dial, new_goal, data, new_req)

        successes += success
        matches += match
        total += 1


        for domain in gen_stats.keys():
            gen_stats[domain][0] += stats[domain][0]
            gen_stats[domain][1] += stats[domain][1]
            gen_stats[domain][2] += stats[domain][2]


        if 'SNG' in filename:
            for domain in gen_stats.keys():
                sng_gen_stats[domain][0] += stats[domain][0]
                sng_gen_stats[domain][1] += stats[domain][1]
                sng_gen_stats[domain][2] += stats[domain][2]

    # BLUE SCORE
    corpus = []
    model_corpus = []
    bscorer = BLEUScorer()

    count_wrong_len = 0
    for dialogue in dialogues:
        data = val_dials[dialogue]
        model_turns, corpus_turns = [], []
        flag = False
        if len(data['sys']) == len(dialogues[dialogue]):
            for idx, turn in enumerate(data['sys']):
                act = data['acts'][idx]  # for different intents
                holding_intents = [a.split('-')[0] for a in act]
                model_turn = dialogues[dialogue][idx]
                if intent in holding_intents:
                    corpus_turns.append([turn])
                    model_turns.append([model_turn])
                    total_turns += 1
                    flag = True
            corpus.extend(corpus_turns)
            model_corpus.extend(model_turns)
        else:
            count_wrong_len += 1
            print('wrong length!!!')

        if flag:
            total_dials +=1

    if count_wrong_len:
        print('count_wrong_len_ratio={}/{}'.format(count_wrong_len, len(dialogues)))
    # Print results
    try:
        BLEU = bscorer.score(model_corpus, corpus)
        MATCHES = (matches / float(total) * 100)
        SUCCESS = (successes / float(total) * 100)
        SCORE = 0.5 * MATCHES + 0.5 * SUCCESS + 100 * BLEU
        print('%s BLEU: %.4f' % (mode, BLEU))
        print('%s Matches: %2.2f%%' % (mode, MATCHES))
        print('%s Success: %2.2f%%' % (mode, SUCCESS))
        print('%s Score: %.4f' % (mode, SCORE))
        print('%s Dialogues: %s' % (mode, total_dials))
        print('%s Evaluated Turns: %s' % (mode, total_turns))
        return BLEU, MATCHES, SUCCESS, SCORE, total
    except:
        print('SCORE ERROR')

def evaluateGeneratedDialogue(dialog, goal, realDialogue, real_requestables):
    """Evaluates the dialogue created by the models.
    First we load the user goal of the dialogue, then for each turn
    generated by the system we look for key-words.
    For the Inform rate we look whether the entity was proposed.
    For the Success rate we look for requestables slots"""
    # for computing corpus success
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    # CHECK IF MATCH HAPPENED
    provided_requestables = {}
    venue_offered = {}
    domains_in_goal = []

    for domain in goal.keys():
        venue_offered[domain] = []
        provided_requestables[domain] = []
        domains_in_goal.append(domain)

    for t, sent_t in enumerate(dialog):
        for domain in goal.keys():
            # for computing success
            if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(domain, realDialogue['log'][t*2 + 1])

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = random.sample(venues, 1)
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0] == ven:
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            # print venues
                            venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            # ATTENTION: assumption here - we didn't provide phone or address twice! etc
            for requestable in requestables:
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'train_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        # if name was provided for the user, the match is being done automatically
        if 'info' in realDialogue['goal'][domain]:
            if 'name' in realDialogue['goal'][domain]['info']:
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'


        if domain == 'train':
            if not venue_offered[domain]:
                if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
                    venue_offered[domain] = '[' + domain + '_name]'

    """
    Given all inform and requestable slots
    we go through each domain from the user goal
    and check whether right entity was provided and
    all requestable slots were given to the user.
    The dialogue is successful if that's the case for all domains.
    """
    # HARD EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0
    success = 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1
        else:
            if domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    #rint requests, 'DIFF', requests_real, 'SUCC', success
    return success, match, stats

def evaluateRealDialogue(dialog, filename):
    """Evaluation of the real dialogue.
    First we loads the user goal and then go through the dialogue history.
    Similar to evaluateGeneratedDialogue above."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    # get the list of domains in the goal
    domains_in_goal = []
    goal = {}
    for domain in domains:
        if dialog['goal'][domain]:
            goal = parseGoal(goal, dialog, domain)
            domains_in_goal.append(domain)

    # compute corpus success
    real_requestables = {}
    provided_requestables = {}
    venue_offered = {}
    for domain in goal.keys():
        provided_requestables[domain] = []
        venue_offered[domain] = []
        real_requestables[domain] = goal[domain]['requestable']

    # iterate each turn
    m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
    for t in range(len(m_targetutt)):
        for domain in domains_in_goal:
            sent_t = m_targetutt[t]
            # for computing match - where there are limited entities
            if domain + '_name' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(domain, dialog['log'][t * 2 + 1])

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = random.sample(venues, 1)
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0] == ven:
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            #print venues
                            venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            for requestable in requestables:
                # check if reference could be issued
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                                #return goal, 0, match, real_requestables
                        elif 'train_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable in sent_t:
                        provided_requestables[domain].append(requestable)

    # offer was made?
    for domain in domains_in_goal:
        # if name was provided for the user, the match is being done automatically
        if 'info' in dialog['goal'][domain]:
            if 'name' in dialog['goal'][domain]['info']:
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'

        # if id was not requested but train was found we dont want to override it to check if we booked the right train
        if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
            venue_offered[domain] = '[' + domain + '_name]'

    # HARD (0-1) EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match, success = 0, 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
            #print(goal_venues)
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1

        else:
            if domain + '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            domain_success = 0
            success_stat = 0
            if len(real_requestables[domain]) == 0:
                # check that
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success +=1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    return goal, success, match, real_requestables, stats

# use the open source evaluation for nlg-eval https://github.com/Maluuba/nlg-eval
def evaluateNLG(gen_dials, ref_dialogues):
    hyp_list, ref_list = [], []
    for fname in gen_dials:
        hyp_list.extend(gen_dials[fname]) # list of sentence string
        ref_list.extend([s.strip() for s in ref_dialogues[fname]['sys']]) # list of ref_list, each ref_list is a list of sentence string
    ref_lists = [ref_list] # only put 1 reference

    from nlgeval import NLGEval
    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics(ref_list=ref_lists, hyp_list=hyp_list)
    print(metrics_dict)
    return metrics_dict

def evaluateNLGFile(gen_dials_fpath, ref_dialogues_fpath):
    with open(gen_dials_fpath, 'r') as gen, open(ref_dialogues_fpath, 'r') as ref:
        gen_dials = json.load(gen)
        ref_dialogues = json.load(ref)

    hyp_list, ref_list = [], []
    for fname in gen_dials:
        hyp_list.extend(gen_dials[fname]) # list of sentence string
        ref_list.extend([s.strip() for s in ref_dialogues[fname]['sys']]) # list of ref_list, each ref_list is a list of sentence string
    ref_lists = [ref_list] # only put 1 reference

    from nlgeval import NLGEval
    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics(ref_list=ref_lists, hyp_list=hyp_list)
    print(metrics_dict)
    return metrics_dict

def evaluateNLGFiles(gen_dials_fpaths, ref_dialogues_fpath):
    from nlgeval import NLGEval
    nlgeval = NLGEval()  # loads the models

    with open(ref_dialogues_fpath, 'r') as ref:
        ref_dialogues = json.load(ref)

    for path in gen_dials_fpaths:
        with open(path, 'r') as gen:
            gen_dials = json.load(gen)

        hyp_list, ref_list = [], []
        for fname in gen_dials:
            hyp_list.extend(gen_dials[fname]) # list of sentence string
            ref_list.extend([s.strip() for s in ref_dialogues[fname]['sys']]) # list of ref_list, each ref_list is a list of sentence string
        ref_lists = [ref_list] # only put 1 reference

        metrics_dict = nlgeval.compute_metrics(ref_list=ref_lists, hyp_list=hyp_list)
        print(path)
        print(metrics_dict)

if __name__ == '__main__':
    pass
    # evaluteNLGFiles(gen_dials_fpath='results/bsl_20190510161309/data/test_dials/test_dials_gen.json', ref_dialogues_fpath='data/test_dials.json')