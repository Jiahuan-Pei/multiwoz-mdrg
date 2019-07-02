# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np

from multiwoz import dbPointer
from multiwoz import Delexicalize

from multiwoz.nlp import normalize


np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def delexicaliseReferenceNumber(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if task['goal']['restaurant']:
        if "book" in turn['metadata']['restaurant']:
            if "booked" in turn['metadata']['restaurant']['book']:
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if task['goal']['hotel']:
        if "book" in turn['metadata']['hotel']:
            if "booked" in turn['metadata']['hotel']['book']:
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if task['goal']['train']:
        if "book" in turn['metadata']['train']:
            if "booked" in turn['metadata']['train']['book']:
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector


def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi',u'restaurant',  u'hospital', u'hotel',u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        #print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    #print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        #print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            #d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            #d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    # pp added
    sys_acts = []
    for i in range(1, len(d['acts'])+1):
        a = d['acts']['%s'%i]
        if a == 'No Annotation':
            sys_acts.append(['UNK-UNK'])
        else:
            sys_acts.append(list(a.keys()))
    d_pp['sys_acts'] = sys_acts
    # pp added

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    db = [t['db_pointer'] for t in d_orig['usr_log']]
    bs = [t['belief_summary'] for t in d_orig['sys_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    # for u, d, s, b in zip(usr, db, sys, bs):
    #     dial.append((u, s, d, b))
    # pp added
    acts = d_orig['sys_acts']
    for u, d, s, b, a in zip(usr, db, sys, bs, acts): # pp added
        dial.append((u, s, d, b, a))
    return dial


def createDict(word_freqs):
    sorted_words=sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)

    # # Extra vocabulary symbols
    # _GO = '_GO'
    # EOS = '_EOS'
    # UNK = '_UNK'
    # PAD = '_PAD'
    # extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()

    for item in sorted_words:
        if len(worddict)>DICT_SIZE:
            break
        worddict[item[0]] = item[1]

    return worddict

def loadData():
    data_url = "data/multi-woz/data.json"
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/multi-woz")

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("data/multi-woz")
        zip_ref.close()
        shutil.copy('data/multi-woz/MULTIWOZ2 2/data.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/valListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/testListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/dialogue_acts.json', 'data/multi-woz/')


def createDelexData():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    print('Creating the delex file...')
    # download the data
    loadData()
    
    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = Delexicalize.prepareSlotValuesIndependent()
    # print(dic)
    delex_data = {}

    fin1 = open('data/multi-woz/data.json')
    data = json.load(fin1)

    fin2 = open('data/multi-woz/dialogue_acts.json')
    data2 = json.load(fin2)

    # pp added: our definition of intent, which consists of [domain]-[act_type]
    intent_set = []
    for dname in data2:
        for sys_turn in data2[dname]:
            try:
                keys = list(data2[dname][sys_turn].keys())
                intent_set.extend(keys)
            except:
                pass
                # "No Annotation"
                # print data2[dname][sys_turn]
    intent_set = ['UNK-UNK'] + sorted(list(set(intent_set)))
    with open('data/intents.json', 'w') as outfile:
        json.dump(intent_set, outfile, indent=4)

    def get_domain(dialogue):
        domains=dict()
        for d in dialogue['goal']:
            if len(dialogue['goal'][d])>0:
                domains[d]=dialogue['goal'][d]
        return domains

    # count = 0
    for dialogue_name in tqdm(data):
        # count +=1
        # if count > 5:
        #     break
        dialogue = data[dialogue_name]
        #print dialogue_name
        domains=get_domain(dialogue)

        # pp added
        try:
            dialogue[u'acts'] = data2[dialogue_name.strip('.json')]
        except:
            print(dialogue_name)

        idx_acts = 1

        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])

            words = sent.split()
            sent = Delexicalize.delexicalise(' '.join(words), dic, domains)

            # parsing reference number GIVEN belief state
            sent = delexicaliseReferenceNumber(sent, turn)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent.strip()

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)
                # add booking pointer
                pointer_vector = addBookingPointer(dialogue, turn, pointer_vector)

                #print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
            idx_acts +=1

        delex_data[dialogue_name] = dialogue

    with open('data/multi-woz/delex.json', 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def divideData(data):
    """Given test and validation sets, divide
    the data for three different sets"""
    print('Diving train/test/valid dataset...')
    testListFile = []
    fin = open('data/multi-woz/testListFile.json')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open('data/multi-woz/valListFile.json')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open('data/trainListFile', 'w')

    test_dials = {}
    val_dials = {}
    train_dials = {}
        
    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()
    # count = 0
    for dialogue_name in tqdm(data):
        #print dialogue_name
        # count +=1
        # if count >5:
        #     break
        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['usr'] = []
            dialogue['sys'] = []
            dialogue['db'] = []
            dialogue['bs'] = []
            dialogue['acts'] = []  # pp added
            for turn in dial:
                dialogue['usr'].append(turn[0])
                dialogue['sys'].append(turn[1])
                dialogue['db'].append(turn[2])
                dialogue['bs'].append(turn[3])
                dialogue['acts'].append(list(turn[4]))  # pp added

            if dialogue_name in testListFile:
                test_dials[dialogue_name] = dialogue
            elif dialogue_name in valListFile:
                val_dials[dialogue_name] = dialogue
            else:
                trainListFile.write(dialogue_name + '\n')
                train_dials[dialogue_name] = dialogue

            for turn in dial:
                line = turn[0]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_usr:
                        word_freqs_usr[w] = 0
                    word_freqs_usr[w] += 1

                line = turn[1]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_sys:
                        word_freqs_sys[w] = 0
                    word_freqs_sys[w] += 1

    # save all dialogues
    with open('data/val_dials.json', 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open('data/test_dials.json', 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open('data/train_dials.json', 'w') as f:
        json.dump(train_dials, f, indent=4)

    return word_freqs_usr, word_freqs_sys


def buildDictionaries(word_freqs_usr, word_freqs_sys):
    """Build dictionaries for both user and system sides.
    You can specify the size of the dictionary through DICT_SIZE variable."""
    dicts = []
    worddict_usr = createDict(word_freqs_usr)
    dicts.append(worddict_usr)
    worddict_sys = createDict(word_freqs_sys)
    dicts.append(worddict_sys)

    # reverse dictionaries
    idx2words = []
    for dictionary in dicts:
        dic = {}
        for k,v in dictionary.items():
            dic[v] = k
        idx2words.append(dic)

    with open('data/input_lang.index2word.json', 'w') as f:
        json.dump(idx2words[0], f, indent=2)
    with open('data/input_lang.word2index.json', 'w') as f:
        json.dump(dicts[0], f,indent=2)
    with open('data/output_lang.index2word.json', 'w') as f:
        json.dump(idx2words[1], f,indent=2)
    with open('data/output_lang.word2index.json', 'w') as f:
        json.dump(dicts[1], f,indent=2)


def main():
    print('Create delexicalized dialogues. Get yourself a coffee, this might take a while.')
    delex_data = createDelexData()
    # delex_data=json.load(open('data/multi-woz/delex.json')) # used for debug to save tiem
    # print('Divide dialogues for separate bits - usr, sys, db, bs')
    print('Divide dialogues for separate bits - usr, sys, db, bs, acts')
    word_freqs_usr, word_freqs_sys = divideData(delex_data)
    print('Building dictionaries')
    buildDictionaries(word_freqs_usr, word_freqs_sys)


if __name__ == "__main__":
    main()
