from spade import model_gnn_2 as spade
import streamlit as st
# import streamlit_tags as tags

import json
import os
import cProfile
from pprint import pformat
# from spade.models import SpadeData
# from spade.bros.bros import BrosConfig
import time

from traceback import print_exc
from pprint import pprint
from functools import cache
from argparse import Namespace
import json

from all_function import *

config = spade.GSpadeConfig(n_layers=5,
                            d_gb_hidden=60,
                            d_edge=60,
                            d_hidden=1280,
                            d_head=768,
                            n_attn_head=1,
                            n_labels=len(fields))


weight_path = "./checkpoint/gcn/0009/best_parsed_f1.pt" 

# checkpoint_path="../../spade-rewrite/checkpoint-bros-vnbill/best_score_parse_vnbill.pt"
checkpoint_path="../spade_weight/checkpoint-gcn-vninvoice-13/best_score_parse_gcn.pt"
os.system("clear")
st.set_page_config(layout="wide")

st.header("Trích xuất hóa đơn")
# detector_vietocr = Predictor(config)

image_path="./data/image/original_image/"
# image_path="/home/phung/phung/Anh_Hung/OCR/OCR-invoice/Vietnamese/spade/label_tool/Image_data/invoice_image_27-5/"+data_json[i]['data_id']


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


with st.spinner(text="Loading model"):
    ie_model = spade.GSpadeForIE(config)
    ie_model = ie_model.float()
    ie_model = ie_model.to(device)
    pretrain = torch.load(weight_path, map_location=device)
    result = ie_model.load_state_dict(pretrain)


    detect_model = get_model_doctr()
    recognize_model = get_model_vietocr()
    tablenet_model= get_tablenet_model("./tablenet_checkpoint/invoice_tablenet_05_09.pth.tar")
    st.success("Model loaded")

# with st.spinner(text="Loading tokenizer"):
#     tokenizer = st.experimental_singleton(models.AutoTokenizer)(
#         "vinai/phobert-base", local_files_only=True)
#     st.success("Tokenizer loaded")

upload_methods = ["Từ thư viện trong máy"] #, "Chụp ảnh mới"]
upload_method = st.radio("Phương pháp upload ảnh", upload_methods)

image = st.file_uploader("Upload file")
# if upload_methods.index(upload_method) == 0:
#     image = st.file_uploader("Upload file")
# else:
#     image = st.camera_input("Chụp ảnh")

left, right = st.columns(2)
if image is not None:
    left.image(image)
    submit = left.button("Nhận dạng")
    clear = left.button("clear")
else:
    submit = clear = False

if submit:
    with st.spinner(text="OCR..."):
        ocr_result=ocr_part(image.getvalue(),detect_model,recognize_model,tablenet_model,need_table=True)
        
        img_copy=ocr_result['image'].copy()
        w,h=ocr_result['width'],ocr_result['height']
        bboxes=ocr_result['merged_boxes']
        for box in bboxes:
            img_copy = bounding_box(box[0][0], box[0][1], box[2][0], box[2][1], img_copy)
        st.image(img_copy, caption='Boxed_image')


    with st.spinner("Inferring..."):
        predict_result, rel_s, rel_g= ie_part(ocr_result,ie_model,need_table=True)


        right.code(json.dumps(predict_result, ensure_ascii=False, indent=2))
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
