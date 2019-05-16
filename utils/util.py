'''
Utility functions
'''

import argparse
import cPickle as pkl
import json
import sys
import math
import time
import numpy as np
import torch
import random

# DEFINE special tokens
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3
# detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

def padSequence(tensor, device=default_device):
    pad_token = PAD_token
    tensor_lengths = [len(sentence) for sentence in tensor]
    longest_sent = max(tensor_lengths)
    batch_size = len(tensor)
    padded_tensor = torch.ones((batch_size, longest_sent), dtype=torch.int64, device=device) * pad_token

    # copy over the actual sequences
    for i, x_len in enumerate(tensor_lengths):
        sequence = tensor[i]
        padded_tensor[i, 0:x_len] = sequence[:x_len]

    padded_tensor = torch.as_tensor(padded_tensor, dtype=torch.long, device=device)
    # padded_tensor = torch.LongTensor(padded_tensor)
    return padded_tensor, tensor_lengths


def loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor=None, intent2index=None, device=default_device):
    # Iterate over dialogue
    for idx, (usr, sys, bs, db, acts) in enumerate(
            zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'], val_file['acts'])):
        tensor = [model.input_word2index(word) for word in usr.strip(' ').split(' ')] + [EOS_token]  # model.input_word2index(word)
        input_tensor.append(torch.as_tensor(tensor, dtype=torch.long, device=device))  # .view(-1, 1))
        # input_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1))

        tensor = [model.output_word2index(word) for word in sys.strip(' ').split(' ')] + [EOS_token]
        target_tensor.append(torch.as_tensor(tensor, dtype=torch.long, device=device))  # .view(-1, 1)
        # target_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1)

        bs_tensor.append([float(belief) for belief in bs])
        db_tensor.append([float(pointer) for pointer in db])

        # pp added: mask_i=0 if i_th it contains i_th intent
        if intent2index:
            tensor = torch.ones(len(intent2index), 1)
            # change acts & find index
            intent_type = model.args.intent_type
            if intent_type == 'domain':
                inds = [model.input_intent2index(act.split('-')[0]) for act in acts]
            elif intent_type == 'sysact':
                inds = [model.input_intent2index(act.split('-')[1]) for act in acts]
            elif intent_type == 'domain_act':
                inds = [model.input_intent2index(act) for act in acts] # the index of the chosen intents
            tensor[:][inds] = 0
            mask_tensor.append(torch.as_tensor(tensor, dtype=torch.uint8, device=device))

    return input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor # mask_tensor is a list of [Intent, 1]


#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))


# pp added -- Start
def get_env_info():
    import sys
    print('Python version={}'.format(sys.version))
    print('PyTorch version={}'.format(torch.__version__))

    flag = torch.cuda.is_available()
    print('torch.cuda.is_available()={}'.format(flag))
    if flag:
        from torch.backends import cudnn
        cudnn.enabled = True
        cudnn.benchmark = False # False efficiency decrease; but fix random;
        cudnn.deterministic = True  # if True, the result would keep same; if False, efficiency would be high but results would change slightly
        # os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose which device to use
        # torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor) # be careful if use
        print('torch.cuda.current_device()={}'.format(torch.cuda.current_device()))
        print('torch.cuda.device_count()={}'.format(torch.cuda.device_count()))
        print('torch.cuda.get_device_name(0)={}'.format(torch.cuda.get_device_name(0)))
        print('torch.backends.cudnn.version()={}'.format(cudnn.version()))
        print('torch.version.cuda={}'.format(torch.version.cuda))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


def get_ms():
    return time.time() * 1000

def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def loadDictionaries(mdir):
    # load data and dictionaries
    with open('{}/input_lang.index2word.json'.format(mdir)) as f:
        input_lang_index2word = json.load(f)
    with open('{}/input_lang.word2index.json'.format(mdir)) as f:
        input_lang_word2index = json.load(f)
    with open('{}/output_lang.index2word.json'.format(mdir)) as f:
        output_lang_index2word = json.load(f)
    with open('{}/output_lang.word2index.json'.format(mdir)) as f:
        output_lang_word2index = json.load(f)


    return input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index

# pp added: give a intent_type, return a list of intent names
def loadIntentDictionaries(intent_type='domain', intent_file='../data/intents.json'):
    fin = file(intent_file)
    intents = json.load(fin) # index 0 is UNK-UNK
    intent_list = []
    if intent_type == 'domain':
        intent_list = [intents[0].split('-')[0]] + sorted(list(set([s.split('-')[0] for s in intents[1:]])))
    elif intent_type == 'sysact':
        intent_list = [intents[0].split('-')[1]] + sorted(list(set([s.split('-')[1] for s in intents[1:]])))
    elif intent_type == 'domain_act':
        intent_list = intents
    if intent_list:
        print('intent_list=', intent_list)
        intent2index = {intent:index for index, intent in enumerate(intent_list)} # the first intent should be 'UNK-UNK'
        index2intent = dict(zip(intent2index.values(), intent2index.keys()))
        return intent2index, index2intent
    else:
        return None, None

# TODO: initialise paras of a model in the same way
def init_params(model):
    from torch.nn.init import xavier_uniform_
    for name, param in model.named_parameters():
        print(name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
# pp added -- End

