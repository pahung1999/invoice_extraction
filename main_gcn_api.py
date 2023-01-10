import torch
import spade.models as models
import streamlit as st
import json
from google.cloud import vision
import os
from transformers import AutoConfig
import spade.transforms as transforms
import cProfile
from pprint import pformat
from detect.ocr import *
import spade.transforms as transforms
# from spade.models import SpadeData
# from spade.bros.bros import BrosConfig
import time
from spade.output_convert import *

#Import gcn (chưa edit))
from spade_gcn.model_gnn import SpadeDataset, ProtoGraphConfig, GCNSpade
from traceback import print_exc
from transformers import AutoTokenizer, BatchEncoding
from torch.utils.data import DataLoader
from pprint import pprint
from functools import cache
from argparse import Namespace
from spade_gcn.score_spade import *
from spade_gcn.score import scores
from spade_gcn.spade_inference import post_process
import json
import requests

url="http://127.0.0.1:8000/uploadfile/doctr_vietocr"


# checkpoint_path="../../spade-rewrite/checkpoint-bros-vnbill/best_score_parse_vnbill.pt"
checkpoint_path="../spade_weight/checkpoint-gcn-vninvoice-13/best_score_parse_gcn.pt"
os.system("clear")
st.set_page_config(layout="wide")

st.header("Trích xuất hóa đơn")
# detector_vietocr = Predictor(config)
MAX_POSITION_EMBEDDINGS = 258 * 3
dataset_config = Namespace(max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                            u_text=30,
                            u_dist=120)

@st.cache
def to_json(bboxes,raw_text,h,w,type="XXYY"):
    def XXYY_to_QUAD(bboxes):
        new_box=[]
        for box in bboxes:
            x1,x2,y1,y2=box[0],box[1],box[2],box[3]
            new_box.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

        return new_box
    def XYXY_to_QUAD(bboxes):
        new_box=[]
        for box in bboxes:
            x1,x2,y1,y2=box[0],box[2],box[1],box[3]
            new_box.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

        return new_box


    #Create dict
    data_dict={}
    data_dict["data_id"]="temp"
    data_dict["text"]=raw_text
    data_dict["label"]= None
    data_dict["img_sz"]= {"width":int(w),"height":int(h)}
    if type == "XXYY":
        data_dict["coord"]=XXYY_to_QUAD(bboxes)
    elif type == "XYXY":
        data_dict["coord"]=XYXY_to_QUAD(bboxes)

    with open('./temp_data.jsonl', 'w', encoding='utf8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False)
        json_file.write("\n")
        json.dump(data_dict, json_file, ensure_ascii=False)

def json_to_batch(json_file):
    BERT = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=False)
    
    OVERLAP = 0
    
    # dataset = SpadeDataset(tokenizer, dataset_config, train_data)
    test_data=  "./temp_data.jsonl"
    test_dataset = SpadeDataset(tokenizer, dataset_config, test_data , fields)

    return test_dataset[0:1]

fields = [
    "info.date",
    "info.form",
    "info.serial",
    "info.num",
    "info.sign_date",
    "seller.name",
    "seller.company",
    "seller.tax",
    "seller.tel",
    "seller.address",
    "seller.bank",
    "customer.name",
    "customer.company",
    "customer.tax",
    "customer.tel",
    "customer.address",
    "customer.bank",
    "customer.payment_method",
    "menu.id",
    "menu.description",
    "menu.unit",
    "menu.quantity",
    "menu.unit_price",
    "menu.subtotal",
    "menu.vat_rate",
    "menu.vat",
    "menu.total",
    "total.subtotal",
    "total.vat_rate",
    "total.vat",
    "total.total"]

@st.experimental_singleton
def get_model():
    BERT = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=False)

    max_epoch = 1000
    MAX_POSITION_EMBEDDINGS = 258 * 3
    OVERLAP = 0
    CHECKPOINTDIR = "checkpoint-gcn-vninvoice-13"
    config = ProtoGraphConfig(
        tokenizer="vinai/phobert-base",
        n_layers=5,
        layer_type="rev_gnn",
        rev_gnn_n_groups=2,
        rev_gnn_mul=True,
        head_rl_layer_type='linear',
        # n_head=12,
        d_model=1280 * 2,
        # d_scales=d_scales,
        # self_loops=[[False] * 4] * 30,
        # update_links=[True] * 30,
        u_text=dataset_config.u_text,
        u_dist=dataset_config.u_dist,
        n_labels=len(fields))


    model = GCNSpade(config)
    sd = torch.load(checkpoint_path, map_location="cpu")
    # print(model)
    model.load_state_dict(sd, strict=False)

    return model


with st.spinner(text="Loading model"):
    model = get_model()
    st.success("Model loaded")

with st.spinner(text="Loading tokenizer"):
    tokenizer = st.experimental_singleton(models.AutoTokenizer)(
        "vinai/phobert-base", local_files_only=True)
    st.success("Tokenizer loaded")

upload_methods = ["Từ thư viện trong máy", "Chụp ảnh mới"]
upload_method = st.radio("Phương pháp upload ảnh", upload_methods)


if upload_methods.index(upload_method) == 0:
    image = st.file_uploader("Upload file")
else:
    image = st.camera_input("Chụp ảnh")

left, right = st.columns(2)
if image is not None:
    left.image(image)
    submit = left.button("Nhận dạng")
    clear = left.button("clear")
else:
    submit = clear = False

if submit:
    with st.spinner(text="OCR..."):
        files = {'file': image}
        res = requests.post(url, files=files)
        raw_text,bboxes,h,w= res.json()[0],res.json()[1],res.json()[2],res.json()[3]
        
        g=arrange_bbox(bboxes, type = "XYXY")
        rows = arrange_row(g= g)

        new_raw_text=[]
        new_bboxes=[]
        for row in rows:
            for i in row:
                new_raw_text.append(raw_text[i])
                new_bboxes.append(bboxes[i])
            # st.write([raw_text[i] for i in row])
        for row in rows:
            x=""
            for i in row:
                x=x+" - "+raw_text[i]
            st.write(x)


        # bboxes,raw_text,h,w=ocr_doctr(image.getvalue())
        to_json(bboxes,raw_text,h,w, type = "XYXY")
        batch=json_to_batch("./temp_data.jsonl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if 'labels' in batch:
            batch.pop('labels')
        # print("batch: ",batch)
        batch = BatchEncoding(batch[0])
        # input_data=transforms.from_doctr_gcn(bboxes,raw_text,h,w)


    with st.spinner("Inferring..."):

        output = model(batch)

        rel_s = output.rel[0].argmax(dim=1)[0].detach().cpu()
        rel_g = output.rel[1].argmax(dim=1)[0].detach().cpu()

        classification, has_loop = post_process(tokenizer, rel_s, rel_g, batch,
                                            fields)

        right.code(json.dumps(classification, ensure_ascii=False, indent=2))
    # a=time.time()
    # with st.spinner(text="Extracting features..."):
      
    #     batch = models.preprocess_gcn({
    #         "tokenizer": "vinai/phobert-base",
    #         "fields": fields
    #     },dataset_config, input_data)
        
    #     for (k, v) in batch.items():
    #         print(k, v.shape)

    # with st.spinner("Inferring..."):
    #     output = model(batch)

    #     rel_s = output.rel[0].argmax(dim=1)[0].detach().cpu()
    #     rel_g = output.rel[1].argmax(dim=1)[0].detach().cpu()

    #     classification, has_loop = post_process(tokenizer, rel_s, rel_g, batch,
    #                                             fields)
    # with st.spinner("Post processing..."):
    #     # final_output = models.post_process(
    #     #     tokenizer,
    #     #     relations=output.relations,
    #     #     batch=batch,
    #     #     fields=fields
    #     # )

    #     right.code(json.dumps(classification, ensure_ascii=False, indent=2))
