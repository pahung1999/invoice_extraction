# import torch
# from transformers import AutoTokenizer
from pprint import pprint
# import re
from collections import Counter
from thefuzz import fuzz
import numpy as np




def score_fuzz(gt, pr):
    keys = [key for key in gt.keys()]
    score = {}
    for key in keys:
        try:
            score[key] = fuzz.ratio(gt[key], pr[key]) / 100
        except Exception:
            score[key] = 0
    return score


def get_scores(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) != 0 else 0
    re = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if (pr + re) != 0 else 0
    return pr, re, f1


def group_counter(group):
    txt = ""
    for key in group:
        txt = txt + str(group[key])
    return Counter(sorted("".join(txt).replace(" ", "")))

def get_group_compare_score(gr1, gr2):
    score = 0
    # if gr1.keys() == gr2.keys():
    #    score += 100

    for key in list(set(list(gr1.keys()) + list(gr2.keys()))):
        if key in gr1 and key in gr2:
            # score+=fuzz.ratio(gr1[key],gr2[key])
            if gr1[key] == gr2[key]:
                score += 50
            elif gr1[key] in gr2[key] or gr2[key] in gr1[key]:
                score += 30
            else:
                score += sum((Counter("".join(gr1[key]))
                              & Counter("".join(gr2[key]))).values())

    score += sum((group_counter(gr1) & group_counter(gr2)).values())
    return score

def norm_receipt(val, key):
    val = val.replace(" ", "")
    return val

def get_all_in_one(gt):
    new_gt=[]
    for key1 in gt:
        new_dict={}
        if key1 !="menu":
            for key2 in gt[key1]:
                new_dict[f"{key1}.{key2}"]=gt[key1][key2]
            new_gt.append(new_dict)
        else:
            for product in gt[key1]:
                new_dict={}
                for key2 in product:
                    new_dict[f"{key1}.{key2}"]=product[key2]
                new_gt.append(new_dict)
    return new_gt

#Version chuyên dùng cho parsed đã qua "hậu xử lý"
def score_spade(gt,pr):
    label_stats = {}
    
    gt=get_all_in_one(gt)
    pr=get_all_in_one(pr)
    # menu_gt=gt['menu']
    # menu_pr=pr['menu']
    mat = np.zeros((len(gt), len(pr)), dtype=np.int)
    for i, group1 in enumerate(gt):

        for j, group2 in enumerate(pr):
            mat[i][j] = get_group_compare_score(group1, group2)
    

    #Ghép cặp sản phẩm tương ứng trong gt,pr
    pairs = []
    for _ in range(min(len(gt), len(pr))):
        if np.max(mat) == 0:
            break
        # print(mat)
        x = np.argmax(mat)
        y = int(x / len(pr))
        x = int(x % len(pr))
        mat[y, :] = 0
        mat[:, x] = 0
        pairs.append((y, x))

    # Đếm số lần từng nhãn xuất hiện trong gt, lưu vào label_stats[key][1]
    for i in range(len(gt)):
        stat = dict()
        for key in gt[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1
        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][1] += stat[key]

    # Đếm số lần từng nhãn xuất hiện trong pr, lưu vào label_stats[key][2]
    for i in range(len(pr)):
        stat = dict()
        for key in pr[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][2] += stat[key]


    for i, j in pairs:
        # For each group,
        stat = dict()
        for key in set(list(gt[i].keys()) + list(pr[j].keys())):
            if key not in stat:
                stat[key] = 0

        cnt = 0
        for key in gt[i]:
            pr_val_1 = ([norm_receipt(val, key)
                       for val in pr[j][key]] if key in pr[j] else [])
            pr_val=[x for x in pr_val_1 if x != ""]
            # print("pr_val: ",pr_val)
            gt_val_1 = [norm_receipt(val, key) for val in gt[i][key]]
            gt_val=[x for x in gt_val_1 if x != ""]
            # print("gt_val: ",gt_val)
            if pr_val == gt_val:
                stat[key] += 1
                cnt += 1
        # Stat Update
        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][0] += stat[key]

    label_stats["all"] = [0, 0, 0]
    for key in sorted(label_stats):
        if key not in ["all"]:
            for i in range(3):
                label_stats["all"][i] += label_stats[key][i]

    s = dict()
    for key in label_stats:
        tp = label_stats[key][0]
        fp = label_stats[key][2] - tp
        fn = label_stats[key][1] - tp
        s[key] = (tp, fp, fn) + get_scores(tp, fp, fn)
    
    return s["all"][3],s["all"][4],s["all"][5]
