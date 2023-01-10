from thefuzz import fuzz
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from spade.model_postprocessors import (
    get_parsed_labeling,
    get_parsed,
    get_parsed_grouping,
    get_parsed_grouping_v2,
    format_parsed_prettyly,
    remove_stopwords,
    remove_stopwords_2,
)
from tqdm import tqdm
import random
from spade import model_gnn_2 as spade
from pprint import pprint
from spade.model_rulebase import rule_base_extract
from typing import Optional

import os
import cv2
import json
from ocr import *

from tablenet.table import *

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    "menu.product_id",
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
    "total.total",
]

def ocr_part(image_path,detect_model,recognize_model,tablenet_model,need_table=True):

    bboxes, image, h, w = detection_doctr(image_path, detect_model)

    raw_text = recognition_vietocr(image, bboxes, recognize_model)

    g = arrange_bbox(bboxes)
    rows = arrange_row(g=g)
    rows = split_row(rows, bboxes, w, ratio=0.2)

    new_text = []
    new_box = []
    box_to_merge = []
    text_to_merge = []
    for i in range(len(rows)):
        box_row = []
        text_row = []
        for j in rows[i]:
            new_text.append(raw_text[j])
            new_box.append(bboxes[j])
            text_row.append(raw_text[j])
            box_row.append(bboxes[j])
        box_to_merge.append(box_row)
        text_to_merge.append(text_row)

    mapping, merged_boxes, merged_texts = get_mapping(
        box_to_merge, text_to_merge)

    if need_table:
        table_part= table_extraction(image, h, w,tablenet_model)
        table_part = cv2.cvtColor(table_part, cv2.COLOR_BGR2GRAY) 
        column_coords_list=get_column_coords(table_part)
        ocr_result=dict(
                    texts=new_text,
                    bboxes=[convert_xyxy_poly(b) for b in new_box],
                    height=h,
                    width=w,
                    mapping=mapping,
                    merged_boxes=[convert_xyxy_poly(b) for b in merged_boxes],
                    merged_texts=merged_texts,
                    column_coords=column_coords_list,
                    image=image
                )
    else:
        ocr_result=dict(
                    texts=new_text,
                    bboxes=[convert_xyxy_poly(b) for b in new_box],
                    height=h,
                    width=w,
                    mapping=mapping,
                    merged_boxes=[convert_xyxy_poly(b) for b in merged_boxes],
                    merged_texts=merged_texts,
                    image=image
                )
    return ocr_result

def get_num_from_string(text):
    num=""
    for i in range(len(text)):
        if text[i].isdigit():
            num+=text[i]
    if num == "":
        return num
    return int(num)

def get_payment(fields, bboxes, texts):
    
    # if len(menu)==0:
    #     return 0
    
    # #Tính tổng các sản phẩm trong menu
    # total_from_menu=[]
    # for product in menu:
    #     if "total" in product:
    #         total_from_menu.append(get_num_from_string(product["total"]))
    # if total_from_menu==[]:
    #     total_from_menu=[0]
    #Xét các "hàng" chứa nhãn
    total_from_label=[]
    total_labels=["Tổng cộng tiền thanh toán","Total payment","Tiền thanh toán","Tổng tiền thanh toán","Tổng cộng"]
    label_box_id=[]
    label_box_center=[]
    for i in range(len(texts)):
        for total_label in total_labels:
            if fuzz.partial_ratio(total_label,texts[i])>95:
                label_box_id.append(i)
    #Tìm các box cùng hàng "total_labels"
    for i in range(len(texts)):
        for box_id in label_box_id:
            center_height=(bboxes[box_id][0][1]+bboxes[box_id][2][1])/2
            if bboxes[i][0][1]<center_height and bboxes[i][2][1]>center_height:
                if get_num_from_string(texts[i])=="":
                    continue
                total_from_label.append(get_num_from_string(texts[i]))
    
    if total_from_label==[]:
        return 0
    return max(total_from_label)

def get_extraction_result(fields, bboxes,texts, rel_s,column_coords_list = None,gt=False):
    # rel_s = output.relations[0].cpu().tolist()
    # print(len(rel_s[0]))
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
    if gt == True:
        return parsed

#     print("parsed before: ")
#     pprint(parsed)

    payment_rule_based=get_payment(fields, bboxes, texts)
    
    menu_rule_based=[]
    if column_coords_list is not None:
        menu_rule_based=menu_extraction_v3(fields, bboxes,texts, rel_s,column_coords_list)
        # print(("-----------------"))
        # print("Menu: ")
        # pprint(menu_rule_based)

#     Use rule base to fill missing information
    # print("Use rule base to fill missing information: ")
    # pprint(rule_base_result)
    # print("--"*30)
    # print(parsed)


    for k, v in rule_base_result.items():
        group, field = k.split('.')

        if group not in parsed:
            parsed[group] = dict()
        if parsed[group].get(k, None) is None:
            parsed[group][k] = v

    # print("parsed second: ")
    # pprint(parsed)

    
    # parsed = apply_recursively(parsed, str, remove_stopwords)
    parsed = remove_stopwords_2(parsed)
    parsed = format_parsed_prettyly(parsed)
#     print("parsed[menu]: ",parsed["menu"])
#     print("menu_rule_based: ",menu_rule_based)
    
#     if len(parsed["menu"])<len(menu_rule_based):
#         parsed["menu"]=menu_rule_based
    parsed["menu"]=menu_rule_based
    
    
    return parsed


def ie_part(ocr_result,model,need_table=False):
        
    ie_input=spade.parse_input(texts=ocr_result['merged_texts'],
                                bboxes=ocr_result['merged_boxes'],
                                width=ocr_result['width'],
                                height=ocr_result['height'])

    batch = ie_input['batch'].to(device)
    with torch.no_grad():
        ie_output = model(batch)
    
    if need_table:
        parsed=get_extraction_result(fields,ocr_result['merged_boxes'], ocr_result['merged_texts'],ie_output.relations[0].cpu().tolist(), column_coords_list=ocr_result['column_coords'])
    else:
        parsed=get_extraction_result(fields,ocr_result['merged_boxes'], ocr_result['merged_texts'],ie_output.relations[0].cpu().tolist())

    return parsed, ie_output.relations[0].cpu().tolist(), ie_output.relations[1].cpu().tolist()


def get_max_area(table_boundRect):
    max_area=0
    max_coord=[0,0,0,0]
    for x,y,w,h in table_boundRect:
        if w*h>max_area:
            max_area=w*h
            max_coord=[x,y,w,h]
    return max_coord[0],max_coord[1],max_coord[2],max_coord[3]
    

def predict(img_path,model):
    
    orig_image = Image.open(img_path)
    or_w,or_h=orig_image.size
    resize_img=orig_image.resize((1024, 1024))
    test_img = np.array(resize_img.convert('LA').convert("RGB"))
    
    image = TRANSFORM(image = test_img)["image"]
    with torch.no_grad():
        image = image.unsqueeze(0)
        #with torch.cuda.amp.autocast():
        table_out= model(image)
        table_out = torch.sigmoid(table_out)

    #remove gradients
    table_out = (table_out.detach().numpy().squeeze(0).transpose(1,2,0) > 0.5).astype(np.uint8)

    #get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    table_contours = []
    #ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
    #remove bad contours
    for c in contours:

        if cv2.contourArea(c) > 3000:
            table_contours.append(c)

    if len(table_contours) == 0:
        print("No Table detected")

    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)

    #table bounding Box
    table_boundRect.sort()

    x,y,w,h=get_max_area(table_boundRect)
    # for x,y,w,h in table_boundRect:
    new_x=int(x*or_w/1024)
    new_y=int(y*or_h/1024)
    new_w=int(w*or_w/1024)
    new_h=int(h*or_h/1024)  
    
    # return new_x,new_y,new_w,new_h
    orig_image=cv2.imread(img_path)
    table_image = 255*np.ones_like(orig_image)
    table_part=orig_image[new_y:new_y+new_h,new_x:new_x+new_w]
    table_image[new_y:new_y+new_h,new_x:new_x+new_w]=table_part

    return table_image, new_x,new_y,new_w,new_h

#Show img
def plot_img(img,size):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#Vẽ bounding box
def bounding_box(x1,y1,x2,y2,img):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img

#Version 1
def menu_extraction(fields, bboxes,texts, rel_s):

    #Tìm giới hạn trên dưới của đầu cột dựa vào box STT
    head_column_upper_bound=0
    head_column_under_bound=0
    # id_min_height=0
    # id_max_height=0
    for i in range(len(texts)):
        if fuzz.partial_ratio("STT",texts[i]) >90:
            head_column_upper_bound=bboxes[i][0][1]
            head_column_under_bound=bboxes[i][2][1]
        
        if head_column_upper_bound !=0 and fuzz.partial_ratio("No",texts[i]) >95:
            if abs(bboxes[i][0][1]-head_column_under_bound)< (head_column_under_bound-head_column_upper_bound):
                head_column_under_bound=bboxes[i][2][1]
        
    if head_column_upper_bound==0:
        print("Không tìm thấy STT")
        return 0
    #Tìm các giới hạn
    column_lr_bound={} #Giới hạn trái phải của cột
    for i in range(len(texts)):
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        if center_height>=head_column_upper_bound and center_height<=head_column_under_bound:
            if fuzz.partial_ratio("STT",texts[i]) >90:
                column_lr_bound["id"]=[bboxes[i][0][0],bboxes[i][2][0]]
            if fuzz.partial_ratio("Tên hàng hóa",texts[i]) >70 or fuzz.partial_ratio("Nội dung",texts[i]) >70 or fuzz.partial_ratio("Tên thiết bị",texts[i]) >70:
                column_lr_bound["description"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("đơn vị",texts[i]) >70 or fuzz.partial_ratio("ĐVT",texts[i]) >70:
                column_lr_bound["unit"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("Số lượng",texts[i]) >60 or fuzz.partial_ratio("SL",texts[i]) >60:
                column_lr_bound["quantity"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("Đơn giá",texts[i]) >60 or fuzz.partial_ratio("Đơn",texts[i]) >85:
                column_lr_bound["unit_price"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("Amount",texts[i]) >75:
                column_lr_bound["subtotal"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("Thuế suất",texts[i]) >75:
                column_lr_bound["vat_rate"]=[bboxes[i][0][0],bboxes[i][2][0]]
                
            if fuzz.partial_ratio("Tiền Thuế",texts[i])>90 and fuzz.partial_ratio("VAT",texts[i]) >90:
                if "vat_rate" in column_lr_bound and column_lr_bound["vat_rate"][0]==bboxes[i][0][0]:
                    continue
                column_lr_bound["vat"]=[bboxes[i][0][0],bboxes[i][2][0]]

            if fuzz.partial_ratio("Total",texts[i]) >60 or fuzz.partial_ratio("Thành tiền",texts[i]) >60:
                column_lr_bound["total"]=[bboxes[i][0][0],bboxes[i][2][0]]

    # print("column_lr_bound: ",column_lr_bound)
    #Danh sách đầu cột
    head_boxes_in_column={}
    column_keys=[key for key in column_lr_bound.keys()]
    for key in column_keys:
        head_boxes_in_column[key]=[]
    for i in range(len(texts)):
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        for key in column_keys:
                center_line=(center_width+(column_lr_bound[key][0]+column_lr_bound[key][1])/2)/2
                if center_line>=column_lr_bound[key][0] and center_line <=column_lr_bound[key][1] and center_height >head_column_upper_bound and center_height <head_column_under_bound :
                    head_boxes_in_column[key].append(i)

    #Giới hạn trên dưới của cột
    #Tính dựa vào box Total

    column_upper_bound=head_column_under_bound
    column_under_bound=head_column_under_bound
    text_end_column=['Cộng tiền hàng',"Thuế suất", "Tổng cộng", "Không chịu thuế", "CỘNG (TOTAL)"] 
    text_in_total_column=['0','1','2','3','4','5','6','7','8','9','x','=']
    for i in range(len(texts)):
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        if any(fuzz.partial_ratio(x,texts[i])>80 for x in text_end_column) and center_height>column_upper_bound:
                column_under_bound = bboxes[i][0][1]
                break
        if "total" in column_lr_bound:
            center_width_line=(center_width+(column_lr_bound["total"][0]+column_lr_bound["total"][1])/2)/2
            if center_width_line>column_lr_bound["total"][0] and center_width_line>column_lr_bound["total"][1] and center_height>column_upper_bound:
                if any(x in texts[i] for x in text_in_total_column) and bboxes[i][2][1] > column_under_bound :
                    column_under_bound = bboxes[i][2][1]
                else:
                    break
    #Danh sách các box nằm trong giới hạn
    menu_boxes={}
    column_keys=[key for key in column_lr_bound.keys()]
    for key in column_keys:
        menu_boxes[key]=[]
    for i in range(len(texts)):
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        if center_height>column_upper_bound and center_height <column_under_bound:
            for key in column_keys:
                center_width_line=(center_width+(column_lr_bound[key][0]+column_lr_bound[key][1])/2)/2
                if center_width_line>=column_lr_bound[key][0] and center_width_line<=column_lr_bound[key][1]:
                    menu_boxes[key].append(i)

                    if bboxes[i][0][0]<column_lr_bound[key][0]:
                        column_lr_bound[key][0]=bboxes[i][0][0]
                    if bboxes[i][2][0]>column_lr_bound[key][1]:
                        column_lr_bound[key][1]=bboxes[i][2][0]
    #Lấy thông tin sản phẩm
    
    #Trường hợp không có sản phẩm
    if len(menu_boxes['id'])==0:
        print("Không tìm thấy sản phẩm")
        return 0

    #Lấy thông tin cận trên cận dưới từng sản phẩm dựa vào STT
    up_un_bouding=[]
    id_boxes=menu_boxes['id']
    up_un_bouding.append(head_column_under_bound)
    for i  in range(len(id_boxes)):
        last_height=up_un_bouding[len(up_un_bouding)-1]
        center_height=(bboxes[id_boxes[i]][0][1]+bboxes[id_boxes[i]][2][1])/2
        if center_height-last_height<0: #Trường hợp có hàng ko chứa STT chen giữa
            up_un_bouding.pop()
            center_height_last=(bboxes[id_boxes[i-1]][0][1]+bboxes[id_boxes[i-1]][2][1])/2
            up_un_bouding.append((center_height+center_height_last)/2)
            last_height=up_un_bouding[len(up_un_bouding)-1]
        up_un_bouding.append(center_height+(center_height-last_height))
    #Lấy thông tin từ các box thuộc menu_boxes
    menu=[]
    for i in range(len(up_un_bouding)-1):
        product={}
        for key in menu_boxes:
            text_of_key=""
            for box_id in  menu_boxes[key]:
                center_height=(bboxes[box_id][0][1]+bboxes[box_id][2][1])/2
                if center_height>up_un_bouding[i] and center_height<up_un_bouding[i+1]:
                    text_of_key+=" "+texts[box_id]
            if text_of_key != "":
                product[key]=text_of_key
        if len(product)>1: #Chỉ lấy trường hợp có nhiều hơn 1 thông tin (tránh chỉ có STT)
            if "description" in product: #Loại dòng kí hiệu ngay dưới dòng STT
                if len(product["description"])<4 or abs(len(product["description"])-len(product['id']))<3:
                    continue
            menu.append(product)

    # print("menu_boxes: ",menu_boxes)
    # for key in menu_boxes:
    #     for i in menu_boxes[key]:
    #         # print(i)
    #         print(f"{key}: {texts[i]}")
    return menu

def get_column_coords(img):
#     plot_img(img,(15,15))
    convert_bin,grey_scale = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    # convert_bin,grey_scale = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    grey_scale = 255-grey_scale

    length = np.array(img).shape[1]//100

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)
    
    
#     plot_img(ver_lines,(15,15))
        
#     print("np.where: ",np.where(ver_lines>0))
    y,x=np.where(ver_lines>0)

    X=np.stack([x,y], axis=1)
    clustering=DBSCAN(eps=2, min_samples=2).fit(X)


    is_line=[False]*(max(clustering.labels_)+1)
    coords=[]
    for i in range(len(is_line)):
        yi = y[i == clustering.labels_]
        xi = x[i == clustering.labels_]
        if np.std(xi) <3:
            is_line[i]=True
            coords.append([xi.min(),yi.min(),xi.max(),yi.max()])
    
    # print("column coord: ",coords)
    # # print(len(is_line))
    # print("is_line: ",is_line)
    def sort_func(x):
        return x[0]
    coords.sort(key=sort_func)

    # print()
    len_column=[abs(coord[1]-coord[3]) for coord in coords]
    # print("len_column: ",len_column)
    max_len= max(len_column)/4
    new_coords=[]
    for i in range(len(coords)):
        if len_column[i]>max_len:
            new_coords.append(coords[i])

    # return np.array(new_coords)
    return np.array(new_coords)



def menu_extraction_v2(fields, bboxes,texts, rel_s,img):

    #Tìm giới hạn trên dưới của đầu cột dựa vào box STT
    head_column_upper_bound=0
    head_column_under_bound=0
    # id_min_height=0
    # id_max_height=0
    for i in range(len(texts)):
        if fuzz.partial_ratio("STT",texts[i]) >90:
            head_column_upper_bound=bboxes[i][0][1]
            head_column_under_bound=bboxes[i][2][1]

        
        if head_column_upper_bound !=0 and fuzz.partial_ratio("No",texts[i]) >95:
            if abs(bboxes[i][0][1]-head_column_under_bound)< (head_column_under_bound-head_column_upper_bound):
                head_column_under_bound=bboxes[i][2][1]
    head_column_quarter=(head_column_under_bound-head_column_upper_bound)/3
    if head_column_upper_bound==0:
        print("Không tìm thấy STT")
        return 0
    #Tìm các giới hạn
    column_coords_list=get_column_coords(img)
    
    column_bound=[x[0] for x in column_coords_list]
    # column_bound.sort()
    print("column_bound: ",column_bound)
    column_upper_bound=head_column_under_bound
    column_under_bound=min([x[3] for x in column_coords_list])

    column_lr_bound={} #Giới hạn trái phải của cột
    for i in range(len(texts)):
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        if fuzz.partial_ratio("Số tiền (Amount)",texts[i])>95:
            continue
        for j in range(len(column_bound)-1):
            if center_height>=head_column_upper_bound and center_height<=head_column_under_bound and center_width>=column_bound[j] and center_width<=column_bound[j+1]:
                column_range=[column_bound[j],column_bound[j+1]]
                # print(f"texts{i}: ",texts[i])
                if fuzz.partial_ratio("STT",texts[i]) >90:
                    column_lr_bound["id"]=column_range
                if fuzz.partial_ratio("Tên hàng hóa",texts[i]) >70 or fuzz.partial_ratio("Nội dung",texts[i]) >70 or fuzz.partial_ratio("Tên thiết bị",texts[i]) >70:
                    column_lr_bound["description"]=column_range

                if fuzz.partial_ratio("đơn vị",texts[i]) >70 or fuzz.partial_ratio("ĐVT",texts[i]) >70:
                    column_lr_bound["unit"]=column_range

                if fuzz.partial_ratio("Số lượng",texts[i]) >60 or fuzz.partial_ratio("SL",texts[i]) >60:
                    column_lr_bound["quantity"]=column_range

                if fuzz.partial_ratio("Đơn giá",texts[i]) >60 or fuzz.partial_ratio("Đơn",texts[i]) >85:
                    column_lr_bound["unit_price"]=column_range

                if fuzz.partial_ratio("Amount",texts[i]) >80:
                    column_lr_bound["subtotal"]=column_range

                if fuzz.partial_ratio("Thuế suất",texts[i]) >75:
                    column_lr_bound["vat_rate"]=column_range
                    
                if fuzz.partial_ratio("Tiền Thuế",texts[i])>90 or fuzz.partial_ratio("VAT",texts[i]) >90:
                    if "vat" not in column_lr_bound:
                        column_lr_bound["vat"]=column_range

                if fuzz.partial_ratio("Total",texts[i]) >60 or fuzz.partial_ratio("Thành tiền",texts[i]) >60:
                    column_lr_bound["total"]=column_range

    # print("column_lr_bound: ",column_lr_bound)
    #Danh sách các box nằm trong giới hạn
    menu_boxes={}
    column_keys=[key for key in column_lr_bound.keys()]
    for key in column_keys:
        menu_boxes[key]=[]
    for i in range(len(texts)):
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        if center_height>column_upper_bound and center_height <column_under_bound:
            for key in column_keys:
                # center_width_line=(center_width+(column_lr_bound[key][0]+column_lr_bound[key][1])/2)/2
                if center_width>=column_lr_bound[key][0] and center_width<=column_lr_bound[key][1]:
                    menu_boxes[key].append(i)
    #Lấy thông tin sản phẩm
    # print("menu_boxes: ",menu_boxes)

    
    #Trường hợp không có sản phẩm
    if len(menu_boxes['id'])==0:
        print("Không tìm thấy sản phẩm")
        return 0

    #Lấy thông tin cận trên cận dưới từng sản phẩm dựa vào STT
    up_un_bouding=[]
    id_boxes=menu_boxes['id']
    up_un_bouding.append(head_column_under_bound)
    for i  in range(len(id_boxes)):
        last_height=up_un_bouding[len(up_un_bouding)-1]
        center_height=(bboxes[id_boxes[i]][0][1]+bboxes[id_boxes[i]][2][1])/2
        if center_height-last_height<0: #Trường hợp có hàng ko chứa STT chen giữa
            up_un_bouding.pop()
            center_height_last=(bboxes[id_boxes[i-1]][0][1]+bboxes[id_boxes[i-1]][2][1])/2
            up_un_bouding.append((center_height+center_height_last)/2)
            last_height=up_un_bouding[len(up_un_bouding)-1]
        up_un_bouding.append(center_height+(center_height-last_height))
    #Lấy thông tin từ các box thuộc menu_boxes
    menu=[]
    for i in range(len(up_un_bouding)-1):
        product={}
        for key in menu_boxes:
            text_of_key=""
            for box_id in  menu_boxes[key]:
                center_height=(bboxes[box_id][0][1]+bboxes[box_id][2][1])/2
                if center_height>up_un_bouding[i] and center_height<up_un_bouding[i+1]:
                    text_of_key+=" "+texts[box_id]
            if text_of_key != "":
                product[key]=text_of_key
        if len(product)>1: #Chỉ lấy trường hợp có nhiều hơn 1 thông tin (tránh chỉ có STT)
            if "description" in product: #Loại dòng kí hiệu ngay dưới dòng STT
                if len(product["description"])<4 or abs(len(product["description"])-len(product['id']))<3:
                    continue
            menu.append(product)

    return menu


def menu_extraction_v3(fields, bboxes,texts, rel_s,column_coords_list):

    #Tìm giới hạn trên dưới của đầu cột dựa vào box STT
    head_column_upper_bound=0
    head_column_under_bound=0
    # id_min_height=0
    # id_max_height=0
    for i in range(len(texts)):
        if fuzz.partial_ratio("STT",texts[i]) >90:
            head_column_upper_bound=bboxes[i][0][1]
            head_column_under_bound=bboxes[i][2][1]

        
        if head_column_upper_bound !=0 and fuzz.partial_ratio("No",texts[i]) >95:
            if abs(bboxes[i][0][1]-head_column_under_bound)< (head_column_under_bound-head_column_upper_bound):
                head_column_under_bound=bboxes[i][2][1]
    head_column_quarter=(head_column_under_bound-head_column_upper_bound)/3
    if head_column_upper_bound==0:
        print("Không tìm thấy STT")
        return []
    #Tìm các giới hạn
    # column_coords_list=get_column_coords(img)
    
    column_bound=[x[0] for x in column_coords_list]
    # column_bound.sort()
    print("column_bound: ",column_bound)
    column_upper_bound=head_column_under_bound
    column_under_bound=min([x[3] for x in column_coords_list])

    column_lr_bound={} #Giới hạn trái phải của cột
    for i in range(len(texts)):
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        if fuzz.partial_ratio("Số tiền (Amount)",texts[i])>95:
            continue
        for j in range(len(column_bound)-1):
            if center_height>=head_column_upper_bound and center_height<=head_column_under_bound and center_width>=column_bound[j] and center_width<=column_bound[j+1]:
                column_range=[column_bound[j],column_bound[j+1]]
                # print(f"texts{i}: ",texts[i])
                if fuzz.partial_ratio("STT",texts[i]) >90:
                    column_lr_bound["id"]=column_range
                if fuzz.partial_ratio("Tên hàng hóa",texts[i]) >70 or fuzz.partial_ratio("Nội dung",texts[i]) >70 or fuzz.partial_ratio("Tên thiết bị",texts[i]) >70:
                    column_lr_bound["description"]=column_range

                if fuzz.partial_ratio("đơn vị",texts[i]) >70 or fuzz.partial_ratio("ĐVT",texts[i]) >70:
                    column_lr_bound["unit"]=column_range

                if fuzz.partial_ratio("Số lượng",texts[i]) >60 or fuzz.partial_ratio("SL",texts[i]) >60:
                    column_lr_bound["quantity"]=column_range

                if fuzz.partial_ratio("Đơn giá",texts[i]) >60 or fuzz.partial_ratio("Đơn",texts[i]) >85:
                    column_lr_bound["unit_price"]=column_range

                if fuzz.partial_ratio("Amount",texts[i]) >80:
                    column_lr_bound["subtotal"]=column_range

                if fuzz.partial_ratio("Thuế suất",texts[i]) >75:
                    column_lr_bound["vat_rate"]=column_range
                    
                if fuzz.partial_ratio("Tiền Thuế",texts[i])>90 or fuzz.partial_ratio("VAT",texts[i]) >90:
                    if "vat" not in column_lr_bound:
                        column_lr_bound["vat"]=column_range

                if fuzz.partial_ratio("Total",texts[i]) >60 or fuzz.partial_ratio("Thành tiền",texts[i]) >60:
                    column_lr_bound["total"]=column_range

    # print("column_lr_bound: ",column_lr_bound)
    #Danh sách các box nằm trong giới hạn
    menu_boxes={}
    column_keys=[key for key in column_lr_bound.keys()]
    for key in column_keys:
        menu_boxes[key]=[]
    for i in range(len(texts)):
        center_width=(bboxes[i][0][0]+bboxes[i][2][0])/2
        
        center_height=(bboxes[i][0][1]+bboxes[i][2][1])/2
        if center_height>column_upper_bound and center_height <column_under_bound:
            for key in column_keys:
                # center_width_line=(center_width+(column_lr_bound[key][0]+column_lr_bound[key][1])/2)/2
                if center_width>=column_lr_bound[key][0] and center_width<=column_lr_bound[key][1]:
                    menu_boxes[key].append(i)
    #Lấy thông tin sản phẩm
    # print("menu_boxes: ",menu_boxes)

    
    #Trường hợp không có sản phẩm
    if "id" not in menu_boxes:
        print("Không tìm thấy sản phẩm")
        return []
    else:
        if len(menu_boxes['id'])==0:
            print("Không tìm thấy sản phẩm")
            return []
        

    #Lấy thông tin cận trên cận dưới từng sản phẩm dựa vào STT
    up_un_bouding=[]
    id_boxes=menu_boxes['id']
    up_un_bouding.append(head_column_under_bound)
    for i  in range(len(id_boxes)):
        last_height=up_un_bouding[len(up_un_bouding)-1]
        center_height=(bboxes[id_boxes[i]][0][1]+bboxes[id_boxes[i]][2][1])/2
        if center_height-last_height<0: #Trường hợp có hàng ko chứa STT chen giữa
            up_un_bouding.pop()
            center_height_last=(bboxes[id_boxes[i-1]][0][1]+bboxes[id_boxes[i-1]][2][1])/2
            up_un_bouding.append((center_height+center_height_last)/2)
            last_height=up_un_bouding[len(up_un_bouding)-1]
        up_un_bouding.append(center_height+(center_height-last_height))
    #Lấy thông tin từ các box thuộc menu_boxes
    menu=[]
    for i in range(len(up_un_bouding)-1):
        product={}
        for key in menu_boxes:
            text_of_key=""
            for box_id in  menu_boxes[key]:
                center_height=(bboxes[box_id][0][1]+bboxes[box_id][2][1])/2
                if center_height>up_un_bouding[i] and center_height<up_un_bouding[i+1]:
                    text_of_key+=" "+texts[box_id]
            if text_of_key != "":
                product[key]=text_of_key
        if len(product)>1: #Chỉ lấy trường hợp có nhiều hơn 1 thông tin (tránh chỉ có STT)
            if "description" in product: #Loại dòng kí hiệu ngay dưới dòng STT
                if len(product["description"])<4 or abs(len(product["description"])-len(product['id']))<3:
                    continue
            menu.append(product)

    return menu