from torch import nn, no_grad
import numpy as np
import torch


def get_graph_from_adj(adj):
    i, j = torch.where(adj)
    edge_index = torch.cat([i[None, :], j[None, :]])
    return edge_index


class BoxSorting(nn.Module):

    @no_grad()
    def forward(self, texts, bboxes):
        return dict(texts=texts, bboxes=bboxes)


class CharFeaturizer(nn.Module):

    def __init__(self, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden

    def featurize_single(self, text):
        special_chars = [
            '&', '@', '#', '(', ')', '-', '+', '=', '*', '%', '.', ',', '\\',
            '/', '|', ':'
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

        seq_ft = np.array(seq_ft)
        return seq_ft

    @no_grad()
    def forward(self, texts, bboxes):
        features = []
        masks = []
        for i, text in enumerate(texts):
            features.append(self.featurize_single(text))
            masks.extend([i] * len(text))

        features = torch.tensor(np.concatenate(features))
        masks = torch.tensor(np.array(masks))
        return dict(sequence_features=features, sequence_masks=masks)


class InvoiceFeaturizer(nn.Module):

    @no_grad()
    def forward(self, texts, bboxes):
        special_chars = [
            '&', '@', '#', '(', ')', '-', '+', '=', '*', '%', '.', ',', '\\',
            '/', '|', ':'
        ]

        # character wise
        n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special = [],[],[],[],[],[]

        # token level feature
        for (i, text) in enumerate(texts):
            lower, upper, alpha, spaces, numeric, special = 0, 0, 0, 0, 0, 0
            for char in text:
                if char.islower():
                    lower += 1
                # for upper letters
                if char.isupper():
                    upper += 1
                # for white spaces
                if char.isspace():
                    spaces += 1
                # for alphabetic chars
                if char.isalpha():
                    alpha += 1
                # for numeric chars
                if char.isnumeric():
                    numeric += 1
                if char in special_chars:
                    special += 1

            n_text = len(text)
            n_lower.append(lower / n_text)
            n_upper.append(upper / n_text)
            n_spaces.append(spaces / n_text)
            n_alpha.append(alpha / n_text)
            n_numeric.append(numeric / n_text)
            n_special.append(special / n_text)

        n_text = len(texts)
        n_chars = [[text.count("/") / n_text for text in texts],
                   [text.count(".") / n_text for text in texts],
                   [text.count(",") / n_text for text in texts],
                   [text.count("(") / n_text for text in texts],
                   [text.count(")") / n_text for text in texts],
                   [text.count("0") / n_text for text in texts],
                   [text.count("%") / n_text for text in texts],
                   [text.count("1") / n_text for text in texts],
                   [text.count("www") / n_text for text in texts],
                   [text.count("http") / n_text for text in texts],
                   [text.count(".com") / n_text for text in texts],
                   [text.count(".org") / n_text for text in texts],
                   [text.count(".gmail") / n_text for text in texts],
                   [text.count(".gov") / n_text for text in texts],
                   [text.count(".net") / n_text for text in texts],
                   [text.count(".vn") / n_text for text in texts],
                   [text.count("@") / n_text for text in texts]]

        features = [n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special]
        features += n_chars
        features = [np.array(f)[:, None] for f in features]
        result = np.concatenate(features, axis=-1)
        return dict(invoice_text_features=result)


class BBoxFeaturizer(nn.Module):

    @no_grad()
    def forward(self, bboxes, width, height):
        bboxes = np.array(bboxes) * 1.0
        bboxes[:, :, 0] /= width
        bboxes[:, :, 1] /= height
        return dict(bboxes=torch.tensor(bboxes))


class DistantBasedGraph(nn.Module):

    def __init__(self, output_name="edge_index"):
        super().__init__()
        self.output_name = output_name

    @torch.no_grad()
    def forward(self, bboxes):
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
        return {self.output_name: get_graph_from_adj(adj)}
