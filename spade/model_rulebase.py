import numpy as np
from thefuzz import fuzz
import re


class FunctionNamespace(dict):
    def __call__(self, k):
        def wrapper(f):
            self[k] = f
        return wrapper


rule_post_processors = FunctionNamespace()


def post_process_taxcode(tax):
    return ''.join(re.findall(r'[0-9()\- .,+]', tax)).strip()


def post_process_money(tax):
    return ''.join(re.findall(r'[0-9() .,]', tax)).strip()


rule_post_processors["seller.tax"] = post_process_taxcode
rule_post_processors["seller.tel"] = post_process_taxcode
rule_post_processors["customer.tax"] = post_process_taxcode
rule_post_processors["customer.tel"] = post_process_taxcode
rule_post_processors["total.total"] = post_process_money
rule_post_processors["total.subtotal"] = post_process_money


@rule_post_processors("total.vat_rate")
def post_process_vatrate(tax):
    return ''.join(re.findall(r'\d+ *%', tax)).strip()


def get_same_row(bboxes):
    ctrs = bboxes.mean(axis=1)
    # cxs = ctrs[..., 0]
    cys = ctrs[..., 1]
    maxs = bboxes.max(axis=1)
    mins = bboxes.min(axis=1)
    # w = bboxes[..., 0].max()
    # h = bboxes[..., 1].max()
    # wds = maxs[:, 0] - mins[:, 0]
    hts = maxs[:, 1] - mins[:, 1]
    # dhs = np.abs(hts[:, None] - hts[None, :])
    # dxs = np.abs(cxs[:, None] - cxs[None, :])
    dys = np.abs(cys[:, None] - cys[None, :])
    is_same_row = np.abs(dys) < ((hts[None, :] + hts[:, None]) / 4)
    return is_same_row


def diffrentiate_role(idx_set, texts, bboxes, default_role=1):
    if len(idx_set) == 0:
        return []
    elif len(idx_set) == 1:
        return [default_role]

    ys = [sum([bboxes[i, ...].mean(axis=0)[-1] for i in idx]) / len(idx)
          for idx in idx_set]
    roles = [None for _ in ys]
    min_y = min(ys)
    max_y = max(ys)
    roles[ys.index(min_y)] = 0
    roles[ys.index(max_y)] = 1
    return roles


def rule_thead(texts, bboxes, is_same_row):
    idx = []
    n = len(texts)
    for i, text in enumerate(texts):
        if fuzz.partial_ratio("STT", text) > 80:
            idx.append(i)

    for i in range(n):
        if i in idx:
            for j in range(n):
                if is_same_row[i, j] and i != j:
                    idx.append(j)
    return [set(idx)]


def rule_label(texts, bboxes, is_same_row, label):
    idx = []
    n = len(texts)
    for i, text in enumerate(texts):
        if fuzz.partial_ratio(label, text) > 80:
            idx.append([i])

    for i_l in idx:
        i = i_l[0]
        for j in range(n):
            if is_same_row[i, j] and i != j and bboxes[i, 0, 0] < bboxes[j, 0, 0]:
                i_l.append(j)

    ret = [list(set(i_l)) for i_l in idx]
    for r in ret:
        r.sort(key=lambda i: bboxes[i, 0, 0])
    return ret


def rule_base_extract(texts, bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    is_same_row = get_same_row(bboxes)

    labels_idx = dict()
    labels_idx['thead'] = rule_thead(texts, bboxes, is_same_row)
    labels_idx['company'] =\
        rule_label(texts, bboxes, is_same_row, label="t??n ????n v???") + \
        rule_label(texts, bboxes, is_same_row, label="????n v??? mua h??ng") + \
        rule_label(texts, bboxes, is_same_row, label="????n v??? b??n h??ng")
    labels_idx['bank'] = rule_label(
        texts, bboxes, is_same_row, label="s??? t??i kho???n")
    labels_idx['address'] = rule_label(
        texts, bboxes, is_same_row, label="?????a ch???")
    labels_idx['tax'] = rule_label(
        texts, bboxes, is_same_row, label="m?? s??? thu???")
    labels_idx['tel'] = rule_label(
        texts, bboxes, is_same_row, label="??i???n tho???i")
    labels_idx['info.sign_date'] = \
        rule_label(texts, bboxes, is_same_row, label="k?? ng??y") + \
        rule_label(texts, bboxes, is_same_row, label="ng??y k??")
    labels_idx['customer.payment_method'] = rule_label(
        texts, bboxes, is_same_row, label="h??nh th???c thanh to??n")
    labels_idx['total.total'] = rule_label(
        texts, bboxes, is_same_row, label="t???ng ti???n thanh to??n")
    labels_idx['total.subtotal'] = rule_label(
        texts, bboxes, is_same_row, label="c???ng ti???n h??ng")
    labels_idx['total.vat_rate'] = rule_label(
        texts, bboxes, is_same_row, label="thu??? su???t")
    labels_idx['info.form'] = rule_label(
        texts, bboxes, is_same_row, label="m???u s???")
    labels_idx['info.num'] = rule_label(
        texts, bboxes, is_same_row, label="s??? hi???u")
    labels_idx['info.serial'] = rule_label(
        texts, bboxes, is_same_row, label="k?? hi???u")

    labels = {k: [([texts[i] for i in v]) for v in vs]
              for k, vs in labels_idx.items()}

    roles = dict()
    for k in ['address', 'tax', 'company', 'bank', "tel"]:
        roles[k] = diffrentiate_role(labels_idx[k], texts, bboxes)

    result = dict()
    for k in labels_idx:
        if k in roles:
            if 0 in roles[k]:
                seller_idx = roles[k].index(0)
                result[f'seller.{k}'] = labels_idx[k][seller_idx]
            if 1 in roles[k]:
                customer_idx = roles[k].index(1)
                result[f'customer.{k}'] = labels_idx[k][customer_idx]

        elif len(labels_idx[k]) > 0 and '.' in k:
            result[k] = labels_idx[k][0]

    result = {k: ' '.join([texts[i] for i in v])
              for k, v in result.items()}

    for k, v in result.items():
        if k in rule_post_processors:
            result[k] = rule_post_processors[k](v)

    return result


# rule_base_classify(ocr_result['merged_texts'],
#                    np.array(ocr_result['merged_boxes']))
