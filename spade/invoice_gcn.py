import numpy as np
import networkx as nx
from torch import nn
from torch_geometric import nn as gnn
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, List
import torch
import json
from argparse import Namespace
from spade.rev_gnn import GroupAddRev
from spade.vocab_embedding import (
    get_vocab_embeds,
    get_vocab_from_file,
    get_vocab_idss,
)


@dataclass
class GSpadeConfig:
    d_head: int = 30
    d_hidden: int = 30
    d_gb_hidden: int = 30
    d_edge: int = 40
    n_attn_head: int = 1
    n_layers: int = 3
    n_labels: int = 31
    n_rels: int = 2
    n_groups: int = 2
    gate_paddings: List[int] = (1, 0)


class SpatialGraphBuilder(nn.Module):

    def __init__(self, d_gb_hidden, directed=False, dropout=0.1):
        super().__init__()
        self.score_0 = nn.LazyConv2d(d_gb_hidden // 2, 3, padding=1)
        self.score_1 = nn.LazyConv2d(d_gb_hidden // 2, 3, padding=1)
        self.score = nn.Conv2d(d_gb_hidden, 2, 1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.weights = nn.LazyConv2d(1, 1)
        self.norm = nn.LayerNorm(1)
        self.directed = directed

    def forward(self, bboxes=None, rel=None, edge_limit=None):
        # bbox dim: n * 8
        if rel.dim() == 3:
            rel = rel.unsqueeze(0)

        score_0 = self.score_0(rel)
        score_1 = self.score_1(rel)
        score = torch.cat([score_0, score_1], dim=1)
        score = self.dropout(score)

        weights = self.norm(self.weights(rel)).relu()
        score = score * (weights > 0)
        if not self.directed:
            score = (score + score.transpose(-1, -2)) / 2
        logits = self.score(score).squeeze(0)

        adj = logits.argmax(dim=0)
        i, j = torch.where(adj)
        edge_index = torch.cat([i[None, :], j[None, :]])
        edge_weights = weights[..., i, j]

        return edge_index, edge_weights


class GSpadeNodeEmbedding(nn.Module):

    def __init__(self, d_seq, d_hidden, n_lstm=1, p_lstm_dropout=0):
        super().__init__()
        d_lstm = min(512, d_hidden // 2)
        self.lstm = nn.RNN(d_seq,
                           d_lstm,
                           n_lstm,
                           dropout=p_lstm_dropout,
                           bidirectional=True)
        self.x_proj = nn.LazyLinear(d_hidden - d_lstm * 2)

    def forward(self, x, seqs, masks):
        embeddings = []
        for i in torch.sort(masks.unique()).values:
            seq = seqs[masks == i]
            hidden, _ = self.lstm(seq)
            embeddings.append(hidden.mean(0, keepdim=True))

        x = self.x_proj(x)
        seqs = torch.cat(embeddings, dim=0)
        x = torch.cat([x, seqs], dim=-1)
        return x
