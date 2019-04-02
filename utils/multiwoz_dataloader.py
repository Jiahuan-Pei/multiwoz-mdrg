#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function :
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-03-28
"""
import torch
import nltk, sys
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.util import *
import json

class MultiwozSingleDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, val_file, name, src_word2id, trg_word2id, intent_type=None, intent2index=None):
        """Reads source and target sequences from txt files."""
        self.val_file = val_file
        self.name = name # the name of json dialogue
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.intent2index = intent2index
        self.intent_type = intent_type
        self.device = torch.device('cpu') # detected_device
        self.input_tensor, self.target_tensor, self.bs_tensor, self.db_tensor, self.mask_tensor = self.SingleDialogueJSON2Tensors()
        self.datalen = self.__len__()

    def __getitem__(self, index):  # data for one dialogue file
        """Returns one data pair (source and target)."""
        input_tensor, target_tensor, bs_tensor, db_tensor = \
        self.input_tensor[index], self.target_tensor[index], self.bs_tensor[index], self.db_tensor[index]
        mask_tensor = self.mask_tensor[index] if self.mask_tensor else None
        return input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor

    def __len__(self):
        return len(self.input_tensor)

    def input_word2index(self, index):
        if self.src_word2id.has_key(index):
            return self.src_word2id[index]
        else:
            return UNK_token

    def out_word2index(self, index):
        if self.trg_word2id.has_key(index):
            return self.trg_word2id[index]
        else:
            return UNK_token

    def SingleDialogueJSON2Tensors(self):
        val_file = self.val_file
        input_tensor = []; target_tensor = []; bs_tensor = []; db_tensor = []; mask_tensor = []
        for idx, (usr, sys, bs, db, acts) in enumerate(
                zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'], val_file['acts'])):
            tensor = [self.input_word2index(word) for word in usr.strip(' ').split(' ')] + [EOS_token]  # model.input_word2index(word)
            input_tensor.append(torch.as_tensor(tensor, dtype=torch.long, device=self.device))  # .view(-1, 1))

            tensor = [self.out_word2index(word) for word in sys.strip(' ').split(' ')] + [EOS_token]
            target_tensor.append(torch.as_tensor(tensor, dtype=torch.long, device=self.device))  # .view(-1, 1)
            # target_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1)

            bs_tensor.append([float(belief) for belief in bs])
            db_tensor.append([float(pointer) for pointer in db])

            # pp added: mask_i=0 if i_th it contains i_th intent
            if self.intent2index:
                tensor = torch.ones(len(self.intent2index), 1)
                # change acts & find index
                intent_type = self.intent_type
                if intent_type == 'domain':
                    inds = [self.intent2index[act.split('-')[0]] for act in acts]
                elif intent_type == 'sysact':
                    inds = [self.intent2index[act.split('-')[1]] for act in acts]
                elif intent_type == 'domain_act':
                    inds = [self.intent2index[act] for act in acts]  # the index of the chosen intents
                tensor[:][inds] = 0
                mask_tensor.append(torch.as_tensor(tensor, dtype=torch.uint8, device=self.device))

        return input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor # each one is a list of tensor

def collate_fn(data, device=torch.device('cpu')):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    data[batch, tuple]; each element of the tuple is a list of tensor
    """
    # batch.sort(key=lambda x: len(x[1]), reverse=True)
    has_mask_tensor = True if data[0][-1] is not None else False
    input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor = zip(*data)

    input_tensor, input_lengths = padSequence(input_tensor)
    target_tensor, target_lengths = padSequence(target_tensor)
    bs_tensor = torch.as_tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.as_tensor(db_tensor, dtype=torch.float, device=device)
    mask_tensor = torch.stack(mask_tensor).permute((1, 0, 2)) if has_mask_tensor else None
    # mask_tensor = torch.stack(mask_tensor).permute((1, 0, 2)) if mask_tensor[0] and mask_tensor[0] != [] else None

    # data = input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor
    # if torch.cuda.is_available():
    #     data = [data[i].cuda() if isinstance(data[i], torch.Tensor) else data[i] for i in range(len(data))]
    return input_tensor, input_lengths, target_tensor, target_lengths, bs_tensor, db_tensor, mask_tensor # tensors [batch_size, *]

def DialogueJSON2Tensors(file_path, input_lang_word2index, output_lang_word2index, intent_type, intent2index):
    m = MultiwozSingleDataset(input_lang_word2index, output_lang_word2index, intent_type, intent2index)
    dials = json.load(open(file_path))
    input_tensor = []; target_tensor = []; bs_tensor = []; db_tensor = []; mask_tensor = []
    for name in dials.keys():
        val_file = dials[name]
        s_input_tensor, s_target_tensor, s_bs_tensor, s_db_tensor, s_mask_tensor = m.SingleDialogueJSON2Tensors(val_file)
        input_tensor.append(s_input_tensor)
        target_tensor.append(s_target_tensor)
        bs_tensor.append(s_bs_tensor)
        db_tensor.append(s_db_tensor)
        mask_tensor.append(s_mask_tensor)
    return input_tensor, target_tensor, bs_tensor, db_tensor, mask_tensor

def get_loader(file_path, src_word2id, trg_word2id, intent_type=None, intent2index=None, batch_size=10):
    """Returns data loader for custom dataset.
    """
    dials = json.load(open(file_path))
    dataset_list = []
    for name in dials.keys():
        val_file = dials[name]
        # build a custom dataset
        dataset = MultiwozSingleDataset(val_file, name, src_word2id, trg_word2id, intent_type, intent2index)
        dataset_list.append(dataset)
    datasets = ConcatDataset(dataset_list)
    # data loader for custome dataset
    data_loader = DataLoader(dataset=datasets,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=collate_fn)
    return data_loader

if __name__ == "__main__":
    data_dir = '../multiwoz-moe/data'
    # intent_type = 'domain'
    intent_type = None
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = loadDictionaries(mdir=data_dir)
    intent2index, index2intent = loadIntentDictionaries(intent_type=intent_type, intent_file='{}/intents.json'.format(data_dir)) if intent_type else (None, None)
    file_path = '{}/train_dials.json'.format(data_dir)
    train_loader = get_loader(file_path, input_lang_word2index, output_lang_word2index, intent_type, intent2index)
    for data in train_loader:
        print data

