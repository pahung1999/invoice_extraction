import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from PIL import Image
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
from functools import lru_cache
import matplotlib.pyplot as plt
import cv2
# Sắp xếp box theo thứ tự trái -> phải, trên -> dưới
# g=arrange_bbox(df["img_bboxes"][i])
# rows=arrange_row(g=g)
from vietocr.tool.config import get_config, list_configs
from vietocr.tool.predictor import Predictor


#Show img
def plot_img(img,size=(8,8)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#Vẽ bounding box
def bounding_box(x1,y1,x2,y2,img):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return img


def arrange_bbox(bboxes):
    n = len(bboxes)
    xcentres = [(b[0] + b[2]) // 2 for b in bboxes]
    ycentres = [(b[1] + b[3]) // 2 for b in bboxes]
    heights = [abs(b[1] - b[3]) for b in bboxes]
    width = [abs(b[2] - b[0]) for b in bboxes]

    def is_top_to(i, j):
        result = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 3)
        return result

    def is_left_to(i, j):
        return (xcentres[i] - xcentres[j]) > ((width[i] + width[j]) / 3)

    # <L-R><T-B>
    # +1: Left/Top
    # -1: Right/Bottom
    g = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            if is_left_to(i, j):
                g[i, j] += 10
            if is_left_to(j, i):
                g[i, j] -= 10
            if is_top_to(i, j):
                g[i, j] += 1
            if is_top_to(j, i):
                g[i, j] -= 1
    return g


def arrange_row(bboxes=None, g=None, i=None, visited=None):
    if visited is not None and i in visited:
        return []
    if g is None:
        g = arrange_bbox(bboxes)
    if i is None:
        visited = []
        rows = []
        for i in range(g.shape[0]):
            if i not in visited:
                indices = arrange_row(g=g, i=i, visited=visited)
                visited.extend(indices)
                rows.append(indices)
        return rows
    else:
        indices = [j for j in range(g.shape[0]) if j not in visited]
        indices = [j for j in indices if abs(g[i, j]) == 10 or i == j]
        indices = np.array(indices)
        g_ = g[np.ix_(indices, indices)]
        order = np.argsort(np.sum(g_, axis=1))
        indices = indices[order].tolist()
        indices = [int(i) for i in indices]
        return indices

def split_row(rows,bboxes,w,ratio=0.5):
    xcentres = [(b[0] + b[2]) // 2 for b in bboxes]
    x1x2= [ [b[0],b[2]] for b in bboxes]  
    mean_hight=np.mean( [abs(b[1] - b[3]) for b in bboxes]) 
    new_rows=[]

    # print("mean_hight: ",mean_hight)
    max_width= int(ratio*mean_hight)
    for row in rows:
        new_row=[row[0]]
        for i in range(1,len(row)):
            if abs(x1x2[row[i]][0]-x1x2[row[i-1]][1]) > max_width:
                new_rows.append(new_row)
                new_row=[row[i]]
            else:
                new_row.append(row[i])
        new_rows.append(new_row)
    
    return new_rows

def convert_box_to_XYXY(bboxes,type_in="XXYY"):
            new_bboxes=[]
            for b in bboxes:
                new_bboxes.append([b[0],b[2],b[1],b[3]])
            return new_bboxes

#Merge box,text, get map
def get_mapping(box_to_merge,text_to_merge):
        mapping=[]
        merged_boxes=[]
        merged_texts=[]
        def merge_box(bboxes):
            min_x=min([b[0] for b in bboxes])
            max_x=max([b[2] for b in bboxes])
            min_y=min([b[1] for b in bboxes])
            max_y=max([b[3] for b in bboxes])
            return [min_x,min_y,max_x,max_y]


        for i in range(len(box_to_merge)):
            merged_box=merge_box(box_to_merge[i])
            merged_text=""
            for text in text_to_merge[i]:
                merged_text+=" "+text
            merged_boxes.append(merged_box)
            merged_texts.append(merged_text)
            for j in range(len(box_to_merge[i])):
                mapping.append(i)
        return mapping, merged_boxes,merged_texts
        
# Detect
@lru_cache
def get_model_doctr(arch='db_resnet50'):
    model = detection_predictor(
        arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    return model


def detection_doctr(image, model):

    single_img_doc = DocumentFile.from_images(image)
    result = model(single_img_doc)

    h, w, c = single_img_doc[0].shape
    bboxes = []
    for box in result[0]:
        x1 = int(box[0]*w)
        y1 = int(box[1]*h)
        x2 = int(box[2]*w)
        y2 = int(box[3]*h)
        bboxes.insert(0, [x1, y1, x2, y2])

    return bboxes, single_img_doc[0], h, w


# Recognition
# input box: x1,y1,x2,y2
@lru_cache
# def get_model_vietocr():
#     config = Cfg.load_config_from_name('vgg_seq2seq')
#     # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
#     config['weights'] = 'https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4'
#     config['cnn']['pretrained'] = False
#     if torch.cuda.is_available():
#         config['device'] = 'cuda:0'
#     else:
#         config['device'] = 'cpu'

#     config['predictor']['beamsearch'] = False
#     model = Predictor(config)
#     return model
def get_model_vietocr(config_name="./checkpoint/vietocr/inception_v3_s2s.yml"):
    
    config = get_config(config_name)
    config['device']= 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Predictor(config)
    return model

def recognition_vietocr(image, bboxes, model):
    raw_text = []
    # image = np.frombuffer(image, np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    for box in bboxes:
        # print("image.shape: ",image.shape)
        # print("box: ",box)
        img_box = image[box[1]:box[3], box[0]:box[2]]
        # print("img_box.shape: ",img_box.shape)
        img_box = Image.fromarray(img_box)
        text = model(img_box)[0]
        if text == []:
            raw_text.append("?")
            continue
        raw_text.append(str(text))
    return raw_text





#Conver  rel_s, rel_g to parsed
def get_parsed_labeling(fields,rel_s):
    itc = rel_s[0:len(fields)]
    stc = rel_s[len(fields):]
    labeling=[-1]*len(rel_s[0])
    head_box=[]
    for i in range(len(fields)):
        for j in range(len(itc[i])):
            if itc[i][j]==1:
                head_box.append(j)
                labeling[j]=i
    
    bigbox_mapping=[]
    while head_box != []:
        i=head_box[0]
        head_box.remove(i)
        new_box=[i]
        visited=[i]
        end_box=False
        while not end_box:
            end_box=True
            for j in range(len(stc[i])):
                if stc[i][j]==1:
                    if j in visited:
                        break
                    visited.append(j)
                    new_box.append(j)
                    i=j
                    end_box=False
                    break
        bigbox_mapping.append(new_box)              
    return labeling,bigbox_mapping

def get_parsed_grouping_v2(fields,rel_g,labeling):
    itc = rel_g[0:len(fields)]
    stc = rel_g[len(fields):]
    grouping=[-1]*len(stc)
    head_group=[]
    for i in range(len(stc)):
        for j in range(len(stc)):
            if stc[i][j]==1 and i not in head_group:
                head_group.append(i)
    for k in range(len(head_group)):
        i=head_group[k]
        grouping[i]=k
        for j in range(len(stc)):
            if stc[i][j]==1:
                grouping[j]=k
        
    return grouping

def get_parsed(fields,texts,labeling, grouping,bigbox_mapping):
    parsed={}
    
    fields_rs=["info","seller","customer","total"]
    for i in fields_rs:
        parsed[f"{i}"]={}
    parsed["menu"]=[]
    parsed_menu={}
    for i in grouping:
        if i != -1:
            parsed_menu[f"{i}"]={}
    for bigbox in bigbox_mapping:
        head_id=bigbox[0]
        text=""
        for i in bigbox:
            text+=" " + texts[i]
        label=fields[labeling[head_id]]
        if grouping[head_id] !=-1:
            parsed_menu[f"{grouping[head_id]}"][label]=text
        else:
            for field_rs in fields_rs:
                if field_rs in label:
                    parsed[f"{field_rs}"][label]=text
            
    group_key=[key for key in parsed_menu.keys()]
    for key in group_key:
        parsed["menu"].append(parsed_menu[key])
    return parsed

#Without rel-g
def get_parsed_grouping(fields,coords,labeling,bigbox_mapping,group_head= False): 
    grouping=[-1]*len(coords)
    grouping_head_id=[]
    menudes_bigbox_mapping=[]
    for bigbox in bigbox_mapping:
        head_id=bigbox[0]
        label=fields[labeling[head_id]]
        if label=="menu.description":
            menudes_bigbox_mapping.append(bigbox)
            grouping_head_id.append(bigbox[0])
            
    all_rows=[]
    for bigbox_des in menudes_bigbox_mapping:
        min_y= min([coords[box_id][0][1] for box_id in bigbox_des ])
        max_y= max([coords[box_id][2][1] for box_id in bigbox_des ])
        row=[]
        for bigbox in bigbox_mapping:
            y_center = (coords[bigbox[0]][0][1]+coords[bigbox[0]][2][1])/2
            if y_center>min_y and y_center<max_y:
                row.append(bigbox)
        all_rows.append(row)
    
    for i in range(len(all_rows)):
        for bigbox in all_rows[i]:
            grouping[bigbox[0]]=i
    
    if group_head:
        return grouping,grouping_head_id
    return grouping



def parsed_convert(parsed):
    fields_rs=[key for key in parsed.keys()]
    new_parsed={}
    for field_rs in fields_rs:
        
        if field_rs=="menu":
            field_rs_dict_new=[]
            for menu_dict in parsed[field_rs]:
                menu_dict_new={}
                menu_dict_keys=[key for key in menu_dict.keys()]
                for key in menu_dict_keys:
                    menu_dict_new[key.replace(f"{field_rs}.","")]=menu_dict[key]
                field_rs_dict_new.append(menu_dict_new)
        else:    
            field_rs_dict_new={}
            field_rs_dict=parsed[field_rs]
            field_rs_dict_keys=[key for key in field_rs_dict.keys()]
            for key in field_rs_dict_keys:
                field_rs_dict_new[key.replace(f"{field_rs}.","")]=field_rs_dict[key]
        new_parsed[field_rs]=field_rs_dict_new
    
    return new_parsed
    
    

def merge_json_data(data,ratio=0.5):

    #Get data infomation
    bboxes=[]
    for box in data['coord']:
        bboxes.append([box[0][0],box[0][1],box[2][0],box[2][1]])
    fields=data["fields"]
    raw_text=data['text']
    rel_s=data['label'][0]
    rel_g=data['label'][1]
    w=data['img_sz']['width']
    h=data['img_sz']['height']
    labeling,bigbox_mapping=get_parsed_labeling(fields,rel_s)
    grouping,group_head=get_parsed_grouping(fields,data['coord'],labeling,bigbox_mapping,group_head=True)

    #Sort, Merge box (get merge_mapping) and convert labeling,bigbox_mapping,grouping according to new sort
    g = arrange_bbox(bboxes)
    rows = arrange_row(g=g)
    rows=split_row(rows,bboxes,w,ratio=ratio)
    new_text = []
    new_box = []
    box_to_merge=[]
    text_to_merge=[]
    new_labeling=[]
    new_grouping=[]
    new_sort=[]
    for i in range(len(rows)):
        box_row=[]
        text_row=[]
        for j in rows[i]:
            new_text.append(raw_text[j])
            new_box.append(bboxes[j])
            text_row.append(raw_text[j])
            box_row.append(bboxes[j])
            new_labeling.append(labeling[j])
            new_grouping.append(grouping[j])
            new_sort.append(j)
        box_to_merge.append(box_row)
        text_to_merge.append(text_row)

    merge_mapping, merged_boxes,merged_texts=get_mapping(box_to_merge,text_to_merge)


    new_bigbox_mapping=[]
    for bigbox in bigbox_mapping:
        new_bigbox=[]
        for box in bigbox:
            new_bigbox.append(new_sort.index(box))
        new_bigbox_mapping.append(new_bigbox)



    #convert labeling,bigbox_mapping,grouping according to merged result
    merged_labeling=[-1]*len(merged_boxes)
    merged_grouping=[-1]*len(merged_boxes)
    for i in range(len(new_labeling)):
        label_id=new_labeling[i]
        group_id=new_grouping[i]
        merged_box_id=merge_mapping[i]
        if merged_labeling[merged_box_id]==-1:
            merged_labeling[merged_box_id]=label_id
        if merged_grouping[merged_box_id]==-1:
            merged_grouping[merged_box_id]=group_id

    merged_bigbox_mapping=[]
    bug_check=[] #Kiểm tra có box nào bị thêm 2 lần
    head_bug=[] #Lưu tạo độ của bigbox chứa box
    for bigbox in new_bigbox_mapping:
        new_box=[]
        for box in bigbox:
            if merge_mapping[box] not in new_box:
                if merge_mapping[box] in bug_check:
                    
                    print(f"Lỗi lặp box {merge_mapping[box]}:{merged_texts[merge_mapping[box]]} khi gộp, box này có thể thuộc 2 label khác nhau")
                    bug_box_label=fields[merged_labeling[head_bug[bug_check.index(merge_mapping[box])]]]
                    print(f"Hệ thống sẽ coi như box này thuộc nhãn {bug_box_label}")
                    data_id=data["data_id"]
                    print(f"Ảnh có box lỗi là {data_id}")
                    print('bigbox: ',bigbox)
                else:
                    new_box.append(merge_mapping[box])
                    bug_check.append(merge_mapping[box])
                    head_bug.append(new_box[0])
                    
        if new_box!=[]:
            merged_bigbox_mapping.append(new_box)
    
    #convert merged_boxes to coord
    new_merged_boxes=[]
    for box in merged_boxes:
        new_merged_boxes.append([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
    
    #New rel_s, rel_g
    rel_s=[[0]*len(merged_texts) for i in range(len(merged_texts)+len(fields))]
    rel_g=[[0]*len(merged_texts) for i in range(len(merged_texts)+len(fields))]

    for i in range(len(merged_labeling)):
        if merged_labeling[i] != -1:
            rel_s[merged_labeling[i]][i]=1
    for bigbox in merged_bigbox_mapping:
        if len(bigbox)>1:
            for i in range(1,len(bigbox)):
                rel_s[len(fields)+bigbox[i-1]][bigbox[i]]=1

    merged_group_head=[]
    for head in group_head:
        new_head=new_sort.index(head)
        merged_group_head.append(merge_mapping[new_head])
        
    for i in range(len(merged_grouping)):
        if merged_grouping[i] != -1:
            group_id=merged_grouping[i]
            
            if merged_group_head[group_id]!=i:
                rel_g[len(fields)+merged_group_head[group_id]][i]=1

    data_new=data.copy()
    data_new['text']=merged_texts
    data_new['label']=[rel_s,rel_g]
    data_new['coord']=new_merged_boxes

    return data_new
    # return


def process_result(result: dict) -> dict:
    """tách và lấy value theo dấu ":" 
    Args:
        result_dict (dict)
    Returns:
        dict
    """
    def get_label(text: str):
        tmp = text.split(":")
        tmp = [i.strip() for i in tmp]
        if len(tmp) < 2:
            return tmp[0]
        else:
            return tmp[1]
    for i in result.keys():
        if not isinstance(result.get(i),list):
            for key in result.get(i).keys():
                result.get(i).update({key: get_label(result.get(i).get(key))})
        else:
            for j in range(len(result.get(i))):
                for key in result.get(i)[j].keys():
                    result.get(i)[j].update({key: get_label(result.get(i)[j].get(key))})
    return result


def convert_xyxy_poly(xyxy):
    x0 = xyxy[0]
    x1 = xyxy[2]
    y0 = xyxy[1]
    y1 = xyxy[3]
    tl = [x0, y0]
    tr = [x1, y0]
    bl = [x0, y1]
    br = [x1, y1]
    return [tl, tr, br, bl]

def ocr_part(image_path,detect_model,recognize_model):

    
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

    ocr_result=dict(
                texts=new_text,
                bboxes=[convert_xyxy_poly(b) for b in new_box],
                height=h,
                width=w,
                mapping=mapping,
                merged_boxes=[convert_xyxy_poly(b) for b in merged_boxes],
                merged_texts=merged_texts,
            )
    return ocr_result

# if __name__=="main":
    