from torch import nn
import torch
from torch_geometric import nn as gnn


def get_graph_from_adj(adj):
    i, j = torch.where(adj)
    edge_index = torch.cat([i[None, :], j[None, :]])
    return edge_index


class CELoss(nn.Module):

    def __init__(self, no_prob, yes_prob):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=torch.tensor([no_prob, yes_prob]))

    def forward(self, score, label):
        scores = score
        labels = label
        if scores.dim() == 3:
            scores = scores.unsqueeze(0)
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)
        labels = labels.type(torch.long)

        # In case the score doesn't have label-text node relation
        # Cut off the label parts from the bottom, leaving
        # only node-to-node predictions
        if labels.size(-2) != scores.size(-2):
            n = scores.size(-2)
            labels = labels[..., -n:, :]

        return dict(loss=self.loss(scores, labels))


class GCNBackbone(nn.Module):

    def __init__(self, d_hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(gnn.GCNConv(d_hidden, d_hidden))

        self.act = nn.GELU()

    def create_graph(self, bboxes: torch.Tensor):
        with torch.no_grad():
            bboxes = bboxes.reshape(-1, 4, 2)
            width = bboxes[..., 0][:].max()
            height = bboxes[..., 1][:].max()
            bboxes[..., 0] /= width
            bboxes[..., 1] /= height
            cxy = bboxes.mean(dim=1)
            cx, cy = cxy[..., 0], cxy[..., 1]
            dx = cx[None, :] - cx[:, None]
            dy = cy[None, :] - cy[:, None]
            dst = (dx**2 + dy**2)**(0.5)
            adj_row = torch.abs(dy) <= 0.05
            adj_col = torch.abs(dx) <= 0.05
            adj_dst = dst <= 0.15
            adj = torch.logical_or(adj_row, adj_dst)
            adj = torch.logical_or(adj_col, adj)
        return get_graph_from_adj(adj)

    def forward(self, x, edge_index):
        for (i, layer) in enumerate(self.layers):
            x = layer(
                x=x,
                edge_index=edge_index,
            )
        x = self.act(x)
        return dict(x=x, edge_index=edge_index)


class RNNEmbedding(nn.Module):

    def __init__(self, d_hidden, n_lstm=1, p_lstm_dropout=0):
        super().__init__()
        # d_lstm = min(512, d_hidden // 2)
        d_lstm = d_hidden // 2
        self.proj = nn.LazyLinear(d_hidden)
        self.lstm = nn.RNN(
            d_hidden,
            d_lstm,
            n_lstm,
            dropout=p_lstm_dropout,
            bidirectional=True,
        )

    def forward(self, sequence_features, sequence_masks):
        sequence_features = self.proj(sequence_features)
        masks = sequence_masks
        embeddings = []
        for i in torch.sort(masks.unique()).values:
            seq = sequence_features[masks == i]
            hidden, _ = self.lstm(seq)
            embeddings.append(hidden.mean(0, keepdim=True))

        seqs = torch.cat(embeddings, dim=0)
        return dict(embeddings=seqs)
