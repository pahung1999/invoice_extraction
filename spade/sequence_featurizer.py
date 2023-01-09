import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache


def get_char_sequence_feature(text):
    special_chars = [
        '&',
        '@',
        '#',
        '(',
        ')',
        '-',
        '+',
        '=',
        '*',
        '%',
        '.',
        ',',
        '\\',
        '/',
        '|',
        ':',
    ]

    seq_ft = []
    for char in text:
        char_ft = [0, 0, 0, 0, 0, 0, 0, 0]
        if char.islower():
            char_ft[0] = 1
        else:
            char_ft[0] = 2
        # for white spaces
        if char.isspace():
            char_ft[1] = 1
        # for alphabetic chars
        if char.isalpha():
            char_ft[1] = 2
        # for numeric chars
        if char.isnumeric():
            char_ft[1] = 3
        if char in special_chars:
            char_ft[1] = 4
        if char == '.':
            char_ft[2] = 2
        if char == '@':
            char_ft[3] = 2
        if char == '%':
            char_ft[4] = 2
        if char == ',':
            char_ft[5] = 2
        if char == '0':
            char_ft[6] = 2
        if char == '/':
            char_ft[7] = 2

        seq_ft.append(np.array(char_ft) / len(char_ft))

    return torch.tensor(seq_ft)


def get_long_masks(texts):

    def is_long(text):
        if text.count(' ') <= 1 and len(text) <= 12:
            return 0
        else:
            return 1

    return np.array([is_long(txt) for txt in texts])


@lru_cache
def tokenizer_from_pretrained(name):
    return AutoTokenizer.from_pretrained(name)


@lru_cache
def model_from_pretrained(name):
    return AutoModel.from_pretrained(name)


@torch.no_grad()
def get_selective_sequence_features(texts, backbone='vinai/phobert-base'):
    tokenizer = tokenizer_from_pretrained(backbone)
    model = model_from_pretrained(backbone)
    long_masks = get_long_masks(texts)
    sequence_features = []
    sequence_masks = []
    for i, (long_mask, text) in enumerate(zip(long_masks, texts)):
        if long_mask == 1:
            input_ids = tokenizer(text, return_tensors='pt')['input_ids']
            embedding = model(input_ids).last_hidden_state.squeeze(0).float()
            sequence_features.append(embedding)
            sequence_masks.extend([i] * embedding.shape[0])
        else:
            embedding = get_char_sequence_feature(text)
            # sequence_features.append(torch.tensor(embedding).float())
            sequence_features.append((embedding.clone().detach()).float())
            sequence_masks.extend([i] * embedding.shape[0])

    sequence_masks = torch.tensor(sequence_masks)
    return sequence_features, sequence_masks, torch.tensor(long_masks)
