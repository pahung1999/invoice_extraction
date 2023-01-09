import numpy as np
import networkx as nx
from torch import nn
from torch_geometric import nn as gnn
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import torch
import json
from argparse import Namespace
from spade.rev_gnn import GroupAddRev
from functools import lru_cache
import re
from tqdm import tqdm
from thefuzz import fuzz
import spade.entity_lists as entity_lists
from spade.sequence_featurizer import get_selective_sequence_features


class DictInput(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__dict__ = self

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
            if isinstance(v, list):
                for (i, v_i) in enumerate(v):
                    if isinstance(v_i, torch.Tensor):
                        self[k][i] = v_i.to(device)
        return self


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


@dataclass
class GSpadeOutput:
    scores: torch.Tensor
    relations: torch.Tensor
    loss: Optional[torch.Tensor]


def force_dim_reduce(x, dim):
    while x.dim() > dim:
        x = x.squeeze(0)
    return x


def get_entity_list_id(text, entities, threshold):
    ratios = [fuzz.partial_ratio(text, entity) for entity in entities]
    max_ratio = max(ratios)
    if max_ratio < threshold:
        return 0
    else:
        return ratios.index(max_ratio) + 1


def get_entity_list_ids(texts, entities, threshold):
    ids = [get_entity_list_id(text, entities, threshold) for text in texts]
    return tensorize(ids)


def get_token_edge_index(token_map):
    edge_index = []
    for tkm in token_map:
        edge_index.extend(list(zip(tkm, tkm[1:])))
    return tensorize(edge_index).transpose(0, 1)


def get_char_sequence_features(texts, max_length=100):

    def get_sequence_feature(text):
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
        # seq_ft = [np.array(s) for s in seq_ft]
        # seq_ft = [torch.tensor(s)[None, :] for s in seq_ft]
        # seq_ft = torch.nn.utils.rnn.pack_sequence(seq_ft)
        # seq_ft = seq_ft.flatten()
        # n = seq_ft.shape[0]
        # if n >= max_length:
        #     seq_ft = seq_ft[:max_length]
        # else:
        #     pad = np.zeros(max_length - n)
        #     seq_ft = np.concatenate([seq_ft, pad])
        return seq_ft

    features = []
    masks = []
    for i, text in enumerate(texts):
        features.append(get_sequence_feature(text))
        masks.extend([i] * len(text))

    features = np.concatenate(features)
    masks = np.array(masks)
    return features, masks


def get_text_features(texts):
    """
    gets text features

    Args: texts: List[str]
    Returns: n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special
    """
    # data = df['Object'].tolist()
    special_chars = [
        '&', '@', '#', '(', ')', '-', '+', '=', '*', '%', '.', ',', '\\', '/',
        '|', ':'
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
    n_chars = [
        [text.count("/") / n_text for text in texts],
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
        [text.count("@") / n_text for text in texts],
    ]

    features = [n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special]
    features += n_chars
    features = [np.array(f)[:, None] for f in features]
    result = np.concatenate(features, axis=-1)
    return result


def get_feature_in_textbox(text, city_list):
    feature_lable = [
        "date", "invoice_serial", "invoice_num", "phone", "bank_account",
        "bank_name", "tax", "address", "long_num", "money"
    ]
    text_feature = [0] * len(feature_lable)

    def check_case(case_list, text, check_city=False):
        for i in range(len(case_list)):
            match = 1 if bool(re.search(case_list[i], text)) else 0
            if match == 1:
                if check_city:
                    return i + 1
                break
        return match

    date_case = [r"\d+/\d+/\d+", r"[Nn]gày .* [Tt]háng .* [Nn]ăm .*"]
    text_feature[feature_lable.index("date")] = check_case(date_case, text)

    invoice_serial_case = [r"Ký hiệu.*\d", r"Serial.*\d"]
    text_feature[feature_lable.index("invoice_serial")] = check_case(
        invoice_serial_case, text)

    invoice_num_case = [r"Số.* H[DĐ] .*\d", r"No. .*\d"]
    text_feature[feature_lable.index("invoice_num")] = check_case(
        invoice_num_case, text)

    phone_case = [r"[Đđ]iện thoại.*\d", r"[tT]el .*\d"]
    text_feature[feature_lable.index("phone")] = check_case(phone_case, text)

    bank_account_case = [r"Số tài khoản.*\d"]
    text_feature[feature_lable.index("bank_account")] = check_case(
        bank_account_case, text)

    bank_name_case = [
        r"[nN]gân hàng .{30}", r"bank.{15}", r"\d .*[nN]gân hàng"
    ]
    text_feature[feature_lable.index("bank_name")] = check_case(
        bank_name_case, text)

    tax_case = [r"Mã số thuế.*\d", r"MST.*\d", r"ERC.*\d"]
    text_feature[feature_lable.index("tax")] = check_case(tax_case, text)

    city_case = city_list.copy()
    # text_feature[feature_lable.index("city")] = check_case(city_case,
    #                                                        text,
    #                                                        check_city=True)

    address_case = city_case + [r"Địa chỉ.{30}"]
    text_feature[feature_lable.index("address")] = check_case(
        address_case, text)

    long_num_case = [r"\d{5}"]
    text_feature[feature_lable.index("long_num")] = check_case(
        long_num_case, text)

    money_case = [r"\d+,\d+", r"\d+\.\d+"]
    text_feature[feature_lable.index("money")] = check_case(money_case, text)

    return text_feature


def get_text_features_v2(texts):
    city_list = [
        "An Giang",
        "Bà Rịa – Vũng Tàu",
        "Bạc Liêu",
        "Bắc Giang",
        "Bắc Kạn",
        "Bắc Ninh",
        "Bến Tre",
        "Bình Dương",
        "Bình Định",
        "Bình Phước",
        "Bình Thuận",
        "Cà Mau",
        "Cao Bằng",
        "Cần Thơ",
        "Đà Nẵng",
        "Đắk Lắk",
        "Đắk Nông",
        "Điện Biên",
        "Đồng Nai",
        "Đồng Tháp",
        "Gia Lai",
        "Hà Giang",
        "Hà Nam",
        "Hà Nội",
        "Hà Tĩnh",
        "Hải Dương",
        "Hải Phòng",
        "Hậu Giang",
        "Hòa Bình",
        "Hồ Chí Minh",
        "Hưng Yên",
        "Khánh Hòa",
        "Kiên Giang",
        "Kon Tum",
        "Lai Châu",
        "Lạng Sơn",
        "Lào Cai",
        "Lâm Đồng",
        "Long An",
        "Nam Định",
        "Nghệ An",
        "Ninh Bình",
        "Ninh Thuận",
        "Phú Thọ",
        "Phú Yên",
        "Quảng Bình",
        "Quảng Nam",
        "Quảng Ngãi",
        "Quảng Ninh",
        "Quảng Trị",
        "Sóc Trăng",
        "Sơn La",
        "Tây Ninh",
        "Thái Bình",
        "Thái Nguyên",
        "Thanh Hóa",
        "Thừa Thiên Huế",
        "Tiền Giang",
        "Trà Vinh",
        "Tuyên Quang",
        "Vĩnh Long",
        "Vĩnh Phúc",
        "Yên Bái",
    ]
    feature_lable = [
        "date", "invoice_serial", "invoice_num", "phone", "bank_account",
        "bank_name", "tax", "address", "long_num", "money"
    ]
    texts_feature = [get_feature_in_textbox(text, city_list) for text in texts]
    result = np.array([np.array(x) for x in texts_feature])
    return result


def batch_consine_sim(batch):
    score = torch.einsum("bih,bjh->bij", batch, batch)
    inv_norm = 1 / torch.norm(batch, dim=-1)
    return torch.einsum("bij,bi,bj->bij", score, inv_norm, inv_norm)


def tensorize(x):
    try:
        return torch.tensor(np.array(x))
    except Exception:
        return torch.tensor(x)


def ensure_numpy(x):
    # Convert to numpy, or just stay as numpy
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.as_array(x)


def ensure_nz(x):
    # Ensure x is not zero
    return x + 1e-6 if x == 0 else x


def get_scores(pr, label):
    pr = ensure_numpy(pr)
    label = ensure_numpy(label)
    tp = ((pr == 1) * (label == 1)).sum()
    # tn = ((pr == 0) * (label == 0)).sum()
    fp = ((pr == 1) * (label == 0)).sum()
    fn = ((pr == 0) * (label == 1)).sum()
    return dict(
        # accuracy=(tp + tn) / ensure_nz(tp + tn + fp + fn),
        precision=tp / ensure_nz(tp + fp),
        recall=tp / ensure_nz(tp + fn),
        f1=2 * tp / ensure_nz(2 * tp + fp + fn),
    )


def get_dist(cx, cy):
    dx = cx[None, :] - cx[:, None]
    dy = cy[None, :] - cy[:, None]
    return (dx**2 + dy**2)**(0.5)


def get_relative_features(bboxes):

    def get_relative_features_single(x, y):
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dists = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        ahs = np.arctan(dx / (dy + 1e-6))
        avs = np.arctan(dy / (dx + 1e-6))
        feats = [dx, dy, dists, ahs, avs]
        feats = [f[None, ...] for f in feats]
        return np.concatenate(feats)

    xs = bboxes[:, :, 0]
    ys = bboxes[:, :, 1]
    feats = [
        get_relative_features_single(xs[:, i], ys[:, i]) for i in range(4)
    ]
    return np.concatenate(feats)


def get_box_graph(bboxes, width=None, height=None):
    # [[x1, y1], ... [x4, y4]]
    # Use numpy.ndarray because it has faster indexing for some reasons
    n = bboxes.shape[0]
    xmaxs = bboxes[:, :, 0].max(axis=1)
    xmins = bboxes[:, :, 0].min(axis=1)
    ymaxs = bboxes[:, :, 1].max(axis=1)
    ymins = bboxes[:, :, 1].min(axis=1)
    xcentres = bboxes[:, :, 0].mean(axis=1)
    ycentres = bboxes[:, :, 1].mean(axis=1)
    heights = ymaxs - ymins
    widths = xmaxs - xmins

    dx = xcentres[None, :] - xcentres[:, None]
    dy = ycentres[None, :] - ycentres[:, None]
    tw = np.abs(widths[None, :] + widths[:, None]) / 3
    th = np.abs(heights[None, :] + heights[:, None]) / 3
    dists = np.sqrt(np.power(dx, 2) + np.power(dy, 2)) - 1

    def is_top_to(i, j):
        is_top = dy[j, i] > th[i, j]
        is_on = np.abs(dx[i, j]) < tw[i, j]
        result = is_top and is_on
        return result

    def is_left_to(i, j):
        # is_left = (xcentres[i] - xcentres[j]) > ((widths[i] + widths[j]) / 3)
        # is_to = abs(ycentres[i] - ycentres[j]) < ((heights[i] + heights[j]) / 3)
        is_left = dx[i, j] > tw[i, j]
        is_to = np.abs(dy[i, j]) < th[i, j]
        return is_left and is_to

    horz = np.zeros((n, n), dtype=float)
    vert = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist = dists[i, j]
            if is_left_to(i, j):
                horz[i, j] = dist + dx[i, j]
            if is_top_to(i, j) and dist / height < 0.15:
                vert[i, j] = dist + dy[i, j]

    horz = nx.minimum_spanning_tree(nx.from_numpy_matrix(horz))
    vert = nx.minimum_spanning_tree(nx.from_numpy_matrix(vert))
    horz = nx.to_numpy_array(horz)
    vert = nx.to_numpy_array(vert)

    for (i, j) in zip(*np.where(horz)):
        if xcentres[i] > xcentres[j]:
            horz[j, i] = horz[i, j]
            horz[i, j] = 0
    for (i, j) in zip(*np.where(vert)):
        if ycentres[i] > ycentres[j]:
            vert[j, i] = vert[i, j]
            vert[i, j] = 0

    return horz, vert


class GSpadeLoss(nn.Module):

    def __init__(self, no_prob, yes_prob):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=torch.tensor([no_prob, yes_prob]))

    def forward(self, scores, labels):
        if scores.dim() == 3:
            scores = scores.unsqueeze(0)
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)
        labels = labels.type(torch.long)

        return self.loss(scores, labels)


class SpatialGraphBuilder(nn.Module):

    def __init__(self, d_hidden=8, edge_dim=None):
        super().__init__()
        self.d_hidden = d_hidden
        self.Q = nn.LazyLinear(d_hidden * 2, bias=False)
        self.K = nn.LazyLinear(d_hidden * 2, bias=False)
        self.V = nn.LazyLinear(d_hidden * 2, bias=False)
        self.norm = nn.LayerNorm(2, elementwise_affine=True)
        self.attr = nn.LazyLinear(edge_dim)

    def forward(self, bboxes=None, rel=None, edge_limit=None):
        # bbox dim: n * 8
        bboxes = force_dim_reduce(bboxes, 2)
        n = bboxes.shape[0]
        Q = self.Q(bboxes).reshape(n, self.d_hidden, 2)
        K = self.K(bboxes).reshape(n, self.d_hidden, 2)
        V = self.V(bboxes).reshape(n, self.d_hidden, 2)
        QK = torch.einsum('mhi,nhi->mni', Q, K)
        QK = self.norm(QK)
        score = torch.einsum('mmi,nhi->mni', QK, V)
        score = score + score.transpose(0, 1)
        attr = self.attr(rel)

        # generate graph
        adj = score.argmax(dim=-1)
        i, j = torch.where(adj)
        edge_index = torch.cat([i[None, :], j[None, :]])
        edge_weights = (score[..., 1] - score[..., 0])[i, j]
        edge_attr = attr[i, j, ...]
        # print(edge_attr.shape)

        # limit number of edge
        if edge_limit < edge_index.shape[0]:
            _, limit_index = torch.topk(edge_weights, edge_limit)
            edge_index = force_dim_reduce(edge_index, 2)[:, limit_index]
            edge_weights = force_dim_reduce(edge_weights, 1)[limit_index]
            edge_attr = edge_attr[limit_index, ...]
        return edge_index, edge_weights, edge_attr


class SpatialGraphBuilder4(nn.Module):

    def __init__(self, in_channels=8, edge_dim=8):
        super().__init__()
        self.edge_dim = edge_dim
        self.score_0 = nn.Conv2d(in_channels, edge_dim // 2, 3, padding=1)
        self.score_1 = nn.Conv2d(in_channels, edge_dim // 2, 3, padding=1)
        self.score = nn.Conv2d(edge_dim, 2, 1, padding=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, bboxes=None, rel=None, edge_limit=None):
        # bbox dim: n * 8
        if rel.dim() == 3:
            rel = rel.unsqueeze(0)

        score_0 = self.score_0(rel)
        score_1 = self.score_1(rel)
        score = torch.cat([score_0, score_1], dim=1)
        score = self.dropout(score)
        score = (score + score.transpose(-1, -2)) / 2
        logits = self.score(score).squeeze(0)

        adj = logits.argmax(dim=0)
        i, j = torch.where(adj)
        edge_index = torch.cat([i[None, :], j[None, :]])
        edge_weights = (logits[1, ...] - logits[0, ...])[i, j]
        edge_attr = force_dim_reduce(score[..., i, j], 2)
        edge_attr = edge_attr.transpose(0, 1)

        # edge_limit = rel.shape[-1] * 10
        if edge_limit is not None and edge_limit < edge_index.shape[0]:
            _, limit_index = torch.topk(edge_weights, edge_limit)
            edge_index = force_dim_reduce(edge_index, 2)[:, limit_index]
            edge_weights = force_dim_reduce(edge_weights, 1)[limit_index]
            edge_attr = edge_attr[limit_index, ...]

        return edge_index, edge_weights, edge_attr


class RevGNNLayer(nn.Module):
    # https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/examples/rev_gnn.py
    def __init__(self, config):
        super().__init__()
        d_hidden = config.d_hidden // config.n_groups
        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=True)
        # self.conv = gnn.GatedGraphConv(d_hidden, 1)
        self.conv = gnn.SAGEConv(d_hidden, d_hidden)
        # self.conv = gnn.GATConv(d_hidden,
        #                         d_hidden,
        #                         config.n_attn_head,
        #                         edge_dim=config.d_edge,
        #                         concat=False)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self,
                x,
                edge_index,
                edge_weights=None,
                edge_attr=None,
                dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)


class GSpadeLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pre_layer_norm = nn.LayerNorm(config.d_hidden,
                                           elementwise_affine=True)
        self.act = nn.GELU()
        n_groups = config.n_groups
        conv = RevGNNLayer(config)
        self.conv = GroupAddRev(conv, num_groups=n_groups)
        self.layer_norm = nn.LayerNorm(config.d_hidden,
                                       elementwise_affine=True)
        self.edge_tf = nn.Sequential(
            nn.LayerNorm(config.d_edge, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(config.d_edge, config.d_edge),
        )
        self.edge_norm = nn.LayerNorm(config.d_edge, elementwise_affine=True)

    def forward(self, x, edge_index, edge_weights, edge_attr):
        x_ = x
        x = self.act(self.pre_layer_norm(x))
        # x = self.conv1(x, edge_index)
        x = self.conv(x, edge_index)
        # x = self.gat(x, edge_index, edge_attr)
        # x = self.conv3(x, edge_index)
        x = self.layer_norm(x + x_)
        edge_attr = self.edge_tf(edge_attr) + edge_attr
        edge_attr = self.edge_norm(edge_attr)
        return x, edge_index, edge_weights, edge_attr


class GSpadeModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        d_hidden = config.d_hidden

        self.graph_builder = SpatialGraphBuilder4(20, config.d_edge)
        # self.node_embeddings = GSpadeNodeEmbedding(8, d_hidden)
        self.node_embeddings = GSpadeNodeEmbedding(768, d_hidden)
        # self.input_proj = nn.LazyLinear(d_hidden)
        self.layers = nn.ModuleList()

        for _ in range(config.n_layers):
            # layer = gnn.ResGatedGraphConv(d_hidden, d_hidden)
            layer = GSpadeLayer(config)
            self.layers.append(layer)
            # self.layers.append(nn.LayerNorm(d_hidden, elementwise_affine=True))
            # self.layers.append(nn.GELU())
            # self.layers.append(nn.Dropout(0.1))

        self.act = nn.GELU()

    def forward(self, batch):
        x = self.node_embeddings(
            x=batch.x,
            seqs=batch.sequence_features,
            masks=batch.sequence_masks,
            bank_ids=batch.bank_ids,
        )
        # x = self.input_proj(batch.x)
        edge_index, edge_weights, edge_attr = self.graph_builder(
            bboxes=batch.bboxes,
            rel=batch.relative_features,
            edge_limit=x.shape[0] * 4)
        for layer in self.layers:
            x, edge_index, edge_weights, edge_attr = layer(
                x=x,
                edge_index=edge_index,
                edge_weights=edge_weights,
                edge_attr=edge_attr,
            )
        x = self.act(x)
        return x


class GSpadeModelB(nn.Module):

    def __init__(self, config):
        super().__init__()
        d_hidden = config.d_hidden

        self.graph_builder = SpatialGraphBuilder4(20, config.d_edge)
        self.node_embeddings = GSpadeNodeEmbeddingBeta(d_hidden)
        # self.input_proj = nn.LazyLinear(d_hidden)
        self.layers = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.bbox_proj = nn.Linear(8, d_hidden)

        for _ in range(config.n_layers):
            # layer = gnn.ResGatedGraphConv(d_hidden, d_hidden)
            layer = GSpadeLayer(config)
            self.layers.append(layer)
            attention = nn.MultiheadAttention(d_hidden,
                                              1,
                                              dropout=0.1,
                                              batch_first=True)
            self.attentions.append(attention)
            self.norms.append(nn.LayerNorm(d_hidden))
            # self.layers.append(nn.LayerNorm(d_hidden, elementwise_affine=True))
            # self.layers.append(nn.GELU())
            # self.layers.append(nn.Dropout(0.1))

        self.act = nn.GELU()

    def forward(self, batch):
        x = self.node_embeddings(
            x=batch.x,
            sequences=batch.sequence_features,
            masks=batch.sequence_masks,
            long_masks=batch.long_masks,
        )
        # x = self.input_proj(batch.x)
        edge_index, edge_weights, edge_attr = self.graph_builder(
            bboxes=batch.bboxes,
            rel=batch.relative_features,
            edge_limit=x.shape[0] * 8)
        bboxes = self.bbox_proj(batch.bboxes)
        for i, layer in enumerate(self.layers):
            h_bboxes, _ = self.attentions[i](bboxes, bboxes, bboxes)
            bboxes = self.norms[i](h_bboxes + bboxes)
            # x = self.norms[i](x + h)
            # print(x.shape, bboxes.shape)
            x = x + bboxes
            x, edge_index, edge_weights, edge_attr = layer(
                x=x,
                edge_index=edge_index,
                edge_weights=edge_weights,
                edge_attr=edge_attr,
            )
            # bboxes = torch.einsum('nm,mh->nh', attention_weights, bboxes)
        x = self.act(x)
        return x


class RelationTagger(nn.Module):

    def __init__(self, in_size, hidden_size, n_fields, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(in_size, hidden_size)
        self.tail = nn.Linear(in_size, hidden_size)
        self.field_embeddings = nn.Parameter(torch.rand(n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.rand(2, 1, 1))

    def forward(self, enc, edge_index=None, edge_weights=None):
        enc_head = self.head(enc)
        enc_tail = self.tail(enc)
        enc_head = torch.cat([self.field_embeddings, enc_head], dim=0)

        score_0 = torch.matmul(enc_head,
                               self.W_label_0(enc_tail).transpose(0, 1))
        score_1 = torch.matmul(enc_head,
                               self.W_label_1(enc_tail).transpose(0, 1))

        score = torch.cat(
            [
                score_0.unsqueeze(0),
                score_1.unsqueeze(0),
            ],
            dim=0,
        )
        score = score * self.gate
        return score


class GSpadeForIE(nn.Module):

    def __init__(self, config, embeddings=None):
        super().__init__()
        self.config = config
        self.backbone = GSpadeModelB(config)
        self.relations = nn.ModuleList()
        self.proj = nn.Linear(config.d_hidden, config.d_head)
        for _ in range(2):
            layer = RelationTagger(config.d_head, config.d_head,
                                   config.n_labels)
            self.relations.append(layer)
        self.loss = GSpadeLoss(0.05, 1)

    def forward(self, batch):
        hidden = self.backbone(batch)
        hidden = self.proj(hidden)
        scores = [layer(hidden) for layer in self.relations]
        relations = [score.argmax(dim=0) for score in scores]

        if 'relations' in batch:
            label_relations = batch.relations
            # loss = ([
            #     self.loss(score, label)
            #     for (score, label) in zip(scores, label_relations)
            # ])
            loss = self.loss(scores[0], label_relations[0])
        else:
            loss = None
        return GSpadeOutput(scores=scores, relations=relations, loss=loss)

    def post_process(self, texts, relations, fields, **kwargs):
        n_labels = len(fields)
        rel_s = ensure_numpy(relations[0])
        # rel_g = relations[1]

        itc_s = rel_s[:n_labels, :]
        stc_s = rel_s[n_labels:, :]

        # Remove self link
        for i in range(len(texts)):
            stc_s[i, i] = 0

        classify = {}
        inv_classify = {}
        for (ifield, jtext) in zip(*np.where(itc_s)):
            inv_classify[jtext] = ifield
            classify[ifield] = [jtext]

        visited = [False for _ in texts]

        def visit(i, visited, ifield):
            if visited[i]:
                return
            visited[i] = True
            for j in np.where(stc_s[i, :] != 0)[0]:
                if not visited[j]:
                    classify[ifield].append(j)
                    visit(j, visited, ifield)

        for (itext, jfield) in inv_classify.items():
            visit(itext, visited, jfield)

        classify = {
            fields[k]: ' '.join([texts[i] for i in v])
            for (k, v) in classify.items()
        }
        # for (itext, jtext) in zip(*np.where(stc_s)):
        #     ifield = inv_classify[itext]
        #     classify[ifield] = [jtext]
        # classification = {texts[k]: fields[i] for (k, i) in itc.items()}
        return classify


def get_long_masks(texts):

    def is_long(text):
        if text.count(' ') or len(text) < 12:
            return 0
        else:
            return 1

    return np.array([is_long(txt) for txt in texts])


def parse_input(texts, bboxes, relations=None, width=None, height=None):
    text_features = get_text_features(texts)
    # sequence_features, sequence_masks = get_sequence_features(texts)
    # sequence_features, sequence_masks = get_dense_sequence_embeddings(texts)
    # sequence_features, sequence_masks = None, None
    char_sequence_features, char_sequence_masks, long_masks =\
        get_selective_sequence_features(texts)
    sequence_masks = char_sequence_masks
    sequence_features = char_sequence_features
    bank_ids = get_entity_list_ids(texts, entity_lists.banks, 50)
    bboxes = np.array(bboxes)
    relative_features = get_relative_features(bboxes)
    # print(type(text_features), type(bboxes.reshape(-1, 8)))
    # dense_features = get_dense_embeddings(texts).numpy()
    x = np.concatenate(
        [  # new line
            text_features,  # new line
            bboxes.reshape(-1, 8),
            # dense_features,
        ],
        axis=-1)
    # print(text_features.shape, bboxes.view(-1, 8).shape)
    n_bboxes = bboxes * 1.0
    n_bboxes[..., 0] = n_bboxes[..., 0] * 10000 / width
    n_bboxes[..., 1] = n_bboxes[..., 1] * 10000 / height
    ret = dict(x=x,
               bank_ids=bank_ids,
               bboxes=n_bboxes.reshape(-1, 8),
               sequence_masks=sequence_masks,
               sequence_features=sequence_features,
               long_masks=long_masks,
               relative_features=relative_features)
    if relations is not None:
        ret['relations'] = np.array(relations)

    for (k, v) in ret.items():
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        if isinstance(v, torch.Tensor
                      ) and not k.endswith("ids") and not k.endswith('masks'):
            ret[k] = v.float()
        # if v.dtype == torch.double:
        #     v.type(torch.float)
        # elif v.dtype == torch.long:
        #     v.type(torch.long)
    return dict(texts=texts,
                bboxes=bboxes,
                relations=ret['relations'],
                batch=DictInput(ret))


class GSpadeNodeEmbedding(nn.Module):

    def __init__(self, d_seq, d_hidden, n_lstm=1, p_lstm_dropout=0):
        super().__init__()
        self.clstm = nn.LSTM(8,
                             d_hidden // 2,
                             n_lstm,
                             dropout=p_lstm_dropout,
                             bidirectional=True)
        # self.lstm = nn.LSTM(768,
        #                     d_hidden // 2,
        #                     n_lstm,
        #                     dropout=p_lstm_dropout,
        #                     bidirectional=True)
        self.x_proj = nn.LazyLinear(d_hidden)
        self.bank_embeddings = nn.Embedding(
            len(entity_lists.banks) + 1, d_hidden)
        self.projection = nn.LazyLinear(d_hidden)

    def forward(self, x, seqs, masks, bank_ids, char_sequence_features,
                char_sequence_masks):
        # embeddings = []
        cembeddings = []
        # for i in torch.sort(masks.unique()).values:
        #     seq = seqs[masks == i]
        #     hidden, _ = self.lstm(seq)
        #     embeddings.append(hidden.sum(0, keepdim=True))

        for i in torch.sort(char_sequence_masks.unique()).values:
            seq = char_sequence_features[char_sequence_masks == i]
            hidden, _ = self.clstm(seq)
            cembeddings.append(hidden.sum(0, keepdim=True))

        x = self.x_proj(x)
        bank_embeddings = self.bank_embeddings(bank_ids)
        # seqs = torch.cat(embeddings, dim=0)
        cseqs = torch.cat(cembeddings, dim=0)
        x = torch.cat([x, cseqs, bank_embeddings], dim=-1)
        x = self.projection(x)
        return x


class LSTMReduce(nn.Module):

    def __init__(self,
                 d_input,
                 d_hidden,
                 n_lstm,
                 p_lstm_dropout,
                 reduction='sum'):
        super().__init__()
        self.lstm = nn.LSTM(
            d_input,
            d_hidden // 2,
            n_lstm,
            dropout=p_lstm_dropout,
            bidirectional=True,
        )
        self.reduction = reduction

    def forward(self, features):
        hidden, _ = self.lstm(features)
        if self.reduction == 'sum':
            return hidden.sum(dim=0)
        elif self.reduction == 'mean':
            return hidden.mean(dim=0)


class GSpadeNodeEmbeddingBeta(nn.Module):

    def __init__(self, d_hidden, n_lstm=1, p_lstm_dropout=0):
        super().__init__()
        self.clstm = LSTMReduce(8, d_hidden, n_lstm, p_lstm_dropout)
        self.dlstm = LSTMReduce(768, d_hidden, n_lstm, p_lstm_dropout)
        self.count_projection = nn.LazyLinear(d_hidden)
        self.projection = nn.LazyLinear(d_hidden)

    def forward(self, x, sequences, masks, long_masks):
        sequence_embeddings = []
        for long_mask, sequence in zip(long_masks, sequences):
            if long_mask == 1:
                hidden = self.dlstm(sequence)
            else:
                hidden = self.clstm(sequence)
            sequence_embeddings.append(hidden.unsqueeze(0))

        sequence_embeddings = torch.cat(sequence_embeddings)
        x = self.count_projection(x)
        x = torch.cat([x, sequence_embeddings], dim=-1)
        x = self.projection(x)
        return x


@lru_cache
def get_bert_pretrained(name):
    model = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    for p in model.parameters():
        p.require_grad = False
    return model, tokenizer


def get_dense_embedding(text, to_cpu=False, aggr=True):
    model, tokenizer = get_bert_pretrained("vinai/phobert-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model = model.to(device)
        features = BatchEncoding(features)
        features = features.to(device)
        output = model(**features).last_hidden_state
        if aggr:
            output = output.sum(dim=1)
        output = output.to("cpu")
        # features = features.cpu()
        if to_cpu:
            model = model.to("cpu")
    return output


def get_dense_embeddings(texts):
    embs = [
        get_dense_embedding(text, i == len(texts) - 1)
        for (i, text) in enumerate(texts)
    ]
    return torch.cat(embs, dim=0)


def get_dense_sequence_embeddings(texts):
    embs = []
    masks = []

    for i, text in enumerate(texts):
        embedding = get_dense_embedding(text, i == len(texts) - 1, aggr=False)
        masks.extend([i] * embedding.shape[1])
        embs.append(embedding)

    masks = tensorize(masks)
    embs = torch.cat(embs, dim=1).squeeze(0)
    return embs, masks


class GSpadeDataset(Dataset):

    def __init__(self, config, jsonl):
        super().__init__()
        with open(jsonl, encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        self.raw = data
        self.fields = data[0]["fields"]
        self.nfields = len(self.fields)
        self._cached_length = len(data)
        self.batch = [
            parse_input(
                texts=d['text'],
                bboxes=d['coord'],
                relations=d.get('label', None),
                width=d['img_sz']['width'],
                height=d['img_sz']['height'],
            ) for d in tqdm(data)
        ]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        return self.batch[idx]
