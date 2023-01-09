import numpy as np


def get_vocab_ids(vocab, text, n=50):
    d = len(vocab)
    unk = d + 0
    pad = d + 1
    sep = d + 2
    cls = d + 3
    ids = [vocab.find(char) for char in text]
    attn_mask = [1 for _ in ids]
    ids = [id_ if id_ > 0 else unk for id_ in ids]
    ids = [sep] + ids + [cls]
    attn_mask = [0] + attn_mask + [0]
    pad_length = n - len(ids)
    ids.extend([pad] * pad_length)
    attn_mask.extend([0] * pad_length)
    return ids, attn_mask


def get_vocab_idss(vocab, texts):
    idss = []
    masks = []
    for mask, text in enumerate(texts):
        ids, mask = get_vocab_ids(vocab, text)
        idss.append(ids)
        masks.append(mask)
    idss = np.array(idss)
    masks = np.array(masks)
    return idss, masks


def get_vocab_embed(vocab, text):
    d = len(vocab) + 1
    n = len(text)
    embs = np.zeros((n, d))
    for (i, char) in enumerate(text):
        embs[i, vocab.find(char)] = 1

    return embs


def get_vocab_embeds(vocab, texts):
    embeds = []
    masks = []
    for mask, text in enumerate(texts):
        masks.extend(len(text) * [mask])
        embeds.append(get_vocab_embed(vocab, text))
    embeds = np.concatenate(embeds)
    masks = np.array(masks)
    return embeds, masks


def get_vocab_from_file(file):
    with open(file, encoding="utf-8") as f:
        vocab = list(set(f.read()))
        vocab = list(sorted(vocab))
        vocab = "".join(vocab)
    return vocab
