from torch import nn
import torch


class RelationTagger(nn.Module):

    def __init__(self, d_hidden, n_labels=0):
        super().__init__()
        self.n_labels = n_labels

        self.head = nn.Linear(d_hidden, d_hidden)
        self.tail = nn.Linear(d_hidden, d_hidden)
        self.W_label_0 = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_label_1 = nn.Linear(d_hidden, d_hidden, bias=False)
        if n_labels > 0:
            self.field_embeddings = nn.Parameter(torch.ones(
                n_labels, d_hidden))

    def forward(self, x):
        enc_head = self.head(x)
        enc_tail = self.tail(x)
        if self.n_labels > 0:
            enc_head = torch.cat([self.field_embeddings, enc_head], dim=0)

        tail_0 = self.W_label_0(enc_tail)
        tail_1 = self.W_label_1(enc_tail)
        score_0 = torch.matmul(enc_head, tail_0.transpose(0, 1))
        score_1 = torch.matmul(enc_head, tail_1.transpose(0, 1))

        # Final score
        score = torch.cat(
            [
                score_0.unsqueeze(0),
                score_1.unsqueeze(0),
            ],
            dim=0,
        )
        return dict(score=score)
