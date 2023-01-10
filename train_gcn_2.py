from pprint import pformat
from spade import model_gnn_2 as spade
import transformers
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import random
from os import path
import os
import shutil
import numpy as np
from spade.score_spade import (
    score_fuzz,
    score_spade
)
from spade.model_postprocessors import (
    get_parsed_labeling,
    get_parsed,
    get_parsed_grouping,
    get_parsed_grouping_v2,
    format_parsed_prettyly,
    remove_stopwords,
    remove_stopwords_2,
)

from spade.model_rulebase import rule_base_extract
from typing import Optional


def get_extraction_result(fields, bboxes,texts, rel_s):
    # rel_s = output.relations[0].cpu().tolist()
    # bboxes = ocr_output['merged_boxes']
    # texts = ocr_output['merged_texts']
    invoice_fields = fields
    # def post_process(self, texts, relations, fields, **kwargs):
    # ie_output = app.invoice_extraction_model.post_process(
    #     texts=ocr_result['texts']
    # )
    rule_base_result =\
        rule_base_extract(texts,
                          bboxes)

    labeling, bigbox_mapping = get_parsed_labeling(invoice_fields, rel_s)
    grouping = get_parsed_grouping(
        invoice_fields,
        bboxes,
        labeling,
        bigbox_mapping,
    )
    parsed = get_parsed(
        invoice_fields,
        texts,
        labeling,
        grouping,
        bigbox_mapping,
    )

    
    
    # Use rule base to fill missing information
    # print(rule_base_result)
    # print("---")
    # print(parsed)
    for k, v in rule_base_result.items():
        group, field = k.split('.')
        if group not in parsed:
            parsed[group] = dict()
        if parsed[group].get(field, None) is None:
            parsed[group][k] = v

    # parsed = apply_recursively(parsed, str, remove_stopwords)
    parsed = remove_stopwords_2(parsed)
    parsed = format_parsed_prettyly(parsed)

    return parsed

class Logger:

    def __init__(self, logdir):
        self.logfile = path.join(logdir, "log.txt")

    def get_mode(self):
        if path.isfile(self.logfile):
            return "a"
        else:
            return "w"

    def print_(self, x, pretty=False):
        s = pformat(x) if pretty else str(x)
        print(s)
        with open(self.logfile, self.get_mode(), encoding='utf-8') as f:
            f.write("\n")
            f.write(s)

    def print(self, x):
        self.print_(x, False)

    def pprint(self, x):
        self.print_(x, True)


def get_checkpoint_dir():
    for i in range(1, 1000):
        chk = path.join("checkpoint", "%04d" % i)
        if not path.isdir(chk):
            return chk


@dataclass
class DataConfig:
    train_data: str
    test_data: str
    n_labels: int


writer = SummaryWriter()
# from torch.nn.parallel import DistributedDataParallel as DDP


def log_print(x):
    with open("train.log", "a") as f:
        f.write(str(x))
        f.write("\n")
        print(x)


data = dict(
    sample=DataConfig(
        train_data="data/sample_data/test.jsonl",
        test_data="data/sample_data/test.jsonl",
        n_labels=1,
    ),
    vn_bill=DataConfig(
        train_data="./data/sample_data/train.jsonl",
        test_data="./data/sample_data/test.jsonl",
        n_labels=27,
    ),
    vn_invoice=DataConfig(
        train_data="./data/vietnamese_invoice_GTGT/train_invoice_vn.jsonl",
        test_data="./data/vietnamese_invoice_GTGT/test_invoice_vn.jsonl",
        n_labels=31,
    ),
    vn_invoice_merge=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn.jsonl",
        test_data="./data/vietnamese_invoice_GTGT_merged/test_invoice_vn.jsonl",
        n_labels=31,
    ),
    vn_invoice_full_origin=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn_full_origin.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_invoice_vn_full_origin.jsonl",
        n_labels=31,
    ),
    vn_invoice_labelv2=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn_labelv2.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_invoice_vn_labelv2.jsonl",
        n_labels=25,
    ),
    vn_invoice_labelv3=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn_labelv3.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_invoice_vn_labelv3.jsonl",
        n_labels=31,
    ),
    vn_invoice_labelv3_full=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn_labelv3_full.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_invoice_vn_labelv3.jsonl",
        n_labels=31,
    ),
    vn_invoice_augment=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_augment_invoice.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_augment_invoice.jsonl",
        n_labels=31,
    ),
    vn_invoice_3_data=DataConfig(
        train_data=
        "./data/vietnamese_invoice_GTGT_merged/train_invoice_vn_labelv3_sample.jsonl",
        test_data=
        "./data/vietnamese_invoice_GTGT_merged/test_invoice_vn_labelv3_sample.jsonl",
        n_labels=31,
    ),
    vn_invoice_2labels=DataConfig(
        train_data="./data/vietnamese_invoice_GTGT_2labels/train_v1.jsonl",
        test_data="./data/vietnamese_invoice_GTGT_2labels/test_v1.jsonl",
        n_labels=2,
    ),
    vn_id=DataConfig(
        train_data="./data/sample_data/spade-data/CCCD/train.jsonl",
        test_data="./data/sample_data/spade-data/CCCD/test.jsonl",
        n_labels=7,
    ),
    jp_card=DataConfig(
        train_data="./data/sample_data/spade-data/business_card/train.jsonl",
        test_data="./data/sample_data/spade-data/business_card/test.jsonl",
        n_labels=7,
    ),
    en_card=DataConfig(
        train_data=
        "./data/sample_data/spade-data/eng_card/train_eng_card.jsonl",
        test_data="./data/sample_data/spade-data/eng_card/test_eng_card.jsonl",
        n_labels=7,
    ),
    CCCD_merge_v2=DataConfig(
        train_data="./data/CCCD/CCCD/merge_v4/CCCD_cut_merge_20-60.jsonl",
        test_data="./data/CCCD/CCCD/merge_v4/CCCD_cut_merge_1_20.jsonl",
        n_labels=8,
    ),
    CCCD_non_chip=DataConfig(
        train_data=
        "/home/phung/AnhHung/label_tool/invoice-data.vi.split/CCCD_non_chip/ver1/CCCD_non_chip_1-15.jsonl",
        test_data=
        "/home/phung/AnhHung/label_tool/invoice-data.vi.split/CCCD_non_chip/ver1/CCCD_non_chip_15-20.jsonl",
        n_labels=9,
    ),
    template_37=DataConfig(
        train_data="./data/template_overfit/Template_37_1-10.jsonl",
        test_data="./data/template_overfit/Template_37_1-10.jsonl",
        n_labels=31))

# dataset = "vn_invoice_merge"
# dataset = "vn_invoice_augment"
# dataset="vn_invoice"
dataset="vn_invoice_labelv3_full"
# dataset = "vn_invoice_labelv3"
# dataset = "vn_invoice_full_origin"
# dataset = "sample"
dataset = data[dataset]
num_warmup_steps = 15
max_epoch = 500 + num_warmup_steps
max_epoch = 1200
train_data = spade.GSpadeDataset(None, dataset.train_data)
test_data = spade.GSpadeDataset(None, dataset.test_data)

config = spade.GSpadeConfig(n_layers=5,
                            d_gb_hidden=60,
                            d_edge=60,
                            d_hidden=1280,
                            d_head=768,
                            n_attn_head=1,
                            n_labels=dataset.n_labels)

model = spade.GSpadeForIE(config)
model = model.float()
print(model)

checkpoint_dir = get_checkpoint_dir()
os.makedirs(checkpoint_dir)
logger = Logger(checkpoint_dir)
logger.print("CONFIG")
logger.pprint(config)
logger.print("DATA")
logger.pprint(data)
logger.print(model)
logger.print(model)
for f in ["train_gcn_2.py", "spade"]:
    if path.isdir(f):
        shutil.copytree(f, path.join(checkpoint_dir, f))
    else:
        shutil.copy(f, path.join(checkpoint_dir, f))

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    opt,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_epoch,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
best = dict()
has_new_best = {}
training_finished = False
keys = test_data.fields
best_score_fuzz = 0
best_score_label = {}  # Best từng label
best_score_fuzz_all = {}  # Label của best mean
best_p_r_f=[0,0,0]
for key in keys:
    best_score_label[key] = 0
    best_score_fuzz_all[key] = 0
for e in range(max_epoch):
    precision=[]
    recall=[]
    f1=[]
    for data in tqdm(train_data):
        data = data['batch']
        opt.zero_grad()
        data = data.to(device)
        model = model.to(device)
        out = model(data)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        out.loss.backward()
        opt.step()
        lr_scheduler.step()

    # Score_fuzz

    score_fuzz_all = {}
    for key in keys:
        score_fuzz_all[key] = []
    metric = dict()
    for data in tqdm(test_data):
        data_or = data
        data = data['batch']
        with torch.no_grad():
            data = data.to(device)
            model = model.to(device)
            out = model(data)
        metrics = [
            spade.get_scores(rel, label)
            for (rel, label) in zip(out.relations, data.relations)
        ]
        for i, m in enumerate(metrics):
            for k, v in m.items():
                k = f"{k}_{i}"
                metric[k] = metric.get(k, 0) + v

        predict = model.post_process(
            data_or['texts'],
            out.relations,
            test_data.fields,
        )
        #  bboxes=data_or['bboxes'])
        ground_truth = model.post_process(
            data_or['texts'],
            data_or['relations'],
            test_data.fields,
        )

        predict_v2 = get_extraction_result(fields= test_data.fields, 
                                    bboxes=data_or['bboxes'],
                                    texts=data_or['texts'], 
                                    rel_s=out.relations[0].cpu().tolist())
    
        ground_truth_v2 = get_extraction_result(fields= test_data.fields, 
                                        bboxes=data_or['bboxes'],
                                        texts=data_or['texts'], 
                                        rel_s=data_or['relations'][0].cpu().tolist())
        p,r,f=score_spade(ground_truth_v2,predict_v2)
        precision.append(p)
        recall.append(r)
        f1.append(f)
        # bboxes=data_or['bboxes'])

        # Score of each data
        score_fuzz_ = score_fuzz(ground_truth, predict)
        keys_gt = [key for key in score_fuzz_.keys()]
        for key_gt in keys_gt:
            x = score_fuzz_all[key_gt]
            x.append(score_fuzz_[key_gt])
            score_fuzz_all[key_gt] = x

    if np.mean(precision) > best_p_r_f[0]:
        best_p_r_f[0]=np.mean(precision)
        torch.save(model.state_dict(),
                       path.join(checkpoint_dir, f"best_parsed_precision.pt"))
    if np.mean(recall) > best_p_r_f[1]:
        best_p_r_f[1]=np.mean(recall)
        torch.save(model.state_dict(),
                       path.join(checkpoint_dir, f"best_parsed_recall.pt"))
    if np.mean(f1) > best_p_r_f[2]:
        best_p_r_f[2]=np.mean(f1)
        torch.save(model.state_dict(),
                       path.join(checkpoint_dir, f"best_parsed_f1.pt"))

    # Mean score of epoch each label
    score_fuzz_mean = []
    for key in keys:
        if score_fuzz_all[key] == []:
            continue
        x = score_fuzz_all[key]
        score_fuzz_all[key] = np.mean(x)
        score_fuzz_mean.append(np.mean(x))

        # Get best score of each label
        if score_fuzz_all[key] > best_score_label[key]:
            best_score_label[key] = score_fuzz_all[key]
            torch.save(model.state_dict(),
                       path.join(checkpoint_dir, f"best_score_fuzz_{key}.pt"))

    score_fuzz_mean = np.mean(score_fuzz_mean)
    # Best mean score
    if score_fuzz_mean > best_score_fuzz:
        best_score_fuzz = score_fuzz_mean
        best_score_fuzz_all = score_fuzz_all.copy()
        torch.save(model.state_dict(),
                   path.join(checkpoint_dir, f"best_score_fuzz.pt"))

    for k, v in metric.items():
        metric[k] = v / len(test_data)
        best_key = f"best_{k}"
        old_best = best.get(best_key, -1)
        best[best_key] = max(best.get(best_key, 0), metric[k])
        has_new_best[best_key] = old_best != best[best_key]

    # Test
    data = random.choice(test_data)
    with torch.no_grad():
        batch = data['batch'].to(device)
        model = model.to(device)
        out = model(batch)

    predict = get_extraction_result(fields= test_data.fields, 
                                    bboxes=data['bboxes'],
                                    texts=data['texts'], 
                                    rel_s=out.relations[0].cpu().tolist())
    
    ground_truth = get_extraction_result(fields= test_data.fields, 
                                    bboxes=data['bboxes'],
                                    texts=data['texts'], 
                                    rel_s=data['relations'][0].cpu().tolist())
    # predict = model.post_process(data['texts'],
    #                              out.relations,
    #                              test_data.fields,
    #                              bboxes=data['bboxes'])
    # ground_truth = model.post_process(data['texts'],
    #                                   data['relations'],
    #                                   test_data.fields,
    #                                   bboxes=data['bboxes'])
    logger.print('=' * 40)
    # logger.print(f'GT\t{ground_truth}')
    logger.print(f'GT')
    logger.pprint(ground_truth)
    logger.print('-+' * 20)
    # logger.print(f'PR\t{predict}')
    logger.print(f'PR')
    logger.pprint(predict)
    logger.print('=' * 40)

    logger.print(f"Score_fuzz_mean:{score_fuzz_mean}")
    logger.print(f"Precision: {np.mean(precision)}")
    logger.print(f"Recall: {np.mean(recall)}")
    logger.print(f"f1: {np.mean(f1)}")
    # logger.print('Score_fuzz:')
    # logger.pprint(score_fuzz_all)  # Điểm của epoch
    logger.print('-+' * 20)
    logger.print(f"Best_Score_fuzz:{best_score_fuzz}")
    logger.print(f"Best Precision: {best_p_r_f[0]}")
    logger.print(f"Best Recall: {best_p_r_f[1]}")
    logger.print(f"Best f1: {best_p_r_f[2]}")
    # logger.print('Best_score_fuzz_all:')
    # logger.pprint(best_score_fuzz_all)  # Điểm của best_score_mean
    # logger.print("Best_score_label:")
    # logger.pprint(best_score_label)  # Best của từng label

    logger.print('=' * 40)
    logger.print(f"Epoch {e}")
    logger.pprint(metric)
    logger.pprint(best)
    logger.print(f"checkpoint: {checkpoint_dir}")

    for (best_key, v) in best.items():
        if has_new_best[best_key]:
            torch.save(model.state_dict(),
                       path.join(checkpoint_dir, f"{best_key}.pt"))
    if e % 100 == 0:
        torch.save(model.state_dict(), path.join(checkpoint_dir, "latest.pt"))

    training_state = dict(
        epoch=e,
        optimizer=opt.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
        training_finised=(e == max_epoch - 1),
        best=best,
        metrics=metrics,
        has_new_best=has_new_best,
    )

    torch.save(training_state, path.join(checkpoint_dir, "training_state.pt"))
