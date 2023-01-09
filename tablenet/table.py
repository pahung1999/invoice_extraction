from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from thefuzz import fuzz
import math

TRANSFORM = A.Compose([
                #ToTensor --> Normalize(mean, std)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value = 255,
                ),
                ToTensorV2()
    ])


class DenseNet(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8,10):
            self.densenet_out_2.add_module(str(x), denseNet[x])
        
        self.densenet_out_3.add_module(str(10), denseNet[10])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        out_1 = self.densenet_out_1(x) #torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1) #torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2) #torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3

class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 256,
                        kernel_size = kernels[0], 
                        stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  #[1, 256, 32, 32]
        out = self.upsample_1_table(x) #[1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1) #[1, 640, 64, 64]
        out = self.upsample_2_table(out) #[1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1) #[1, 512, 128, 128]
        out = self.upsample_3_table(out) #[1, 3, 1024, 1024]
        return out
        


class TableNet(nn.Module):
    def __init__(self,encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True):
        super(TableNet, self).__init__()
        
        self.kernels = [(1,1), (2,2), (2,2),(8,8)]
        self.strides = [(1,1), (2,2), (2,2),(8,8)]
        self.in_channels = 512
        

        if encoder == 'densenet':
            self.base_model = DenseNet(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
            self.pool_channels = [512, 256]
            self.in_channels = 1024
            self.kernels = [(1,1), (1,1), (2,2),(16,16)]
            self.strides = [(1,1), (1,1), (2,2),(16,16)]
        
        #common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)

    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out) #[1, 256, 32, 32]
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([1, 1, 1024, 1024])
        return table_out





def get_tablenet_model(pretrained_path = "./gorc/invoice_tablenet.pth.tar"):

    tablenet_model = TableNet(encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True)
    tablenet_model.eval()

    #load checkpoint
    tablenet_model.load_state_dict(torch.load(pretrained_path)['state_dict'])

    return tablenet_model

def get_max_area(table_boundRect):
    max_area=0
    max_coord=[0,0,0,0]
    for x,y,w,h in table_boundRect:
        if w*h>max_area:
            max_area=w*h
            max_coord=[x,y,w,h]
    return max_coord[0],max_coord[1],max_coord[2],max_coord[3]
    


def table_extraction(orig_image, or_h, or_w,model):
    

    color_coverted = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)

    resize_img=pil_image.resize((1024, 1024))
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
    table_image = 255*np.ones_like(orig_image)
    table_part=orig_image[new_y:new_y+new_h,new_x:new_x+new_w]
    table_image[new_y:new_y+new_h,new_x:new_x+new_w]=table_part

    # return table_image, new_x,new_y,new_w,new_h
    return table_image

def get_column_coords(img):
    # convert_bin,grey_scale = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    convert_bin,grey_scale = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    grey_scale = 255-grey_scale

    length = np.array(img).shape[1]//100

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)

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
            coords.append([int(xi.min()),int(yi.min()),int(xi.max()),int(yi.max())])
    
    # print("column coord: ",coords)
    # # print(len(is_line))
    # print("is_line: ",is_line)
    def sort_func(x):
        return x[0]
    coords.sort(key=sort_func)

    len_column=[abs(coord[1]-coord[3]) for coord in coords]
    max_len= max(len_column)/3
    new_coords=[]
    for i in range(len(coords)):
        if len_column[i]>max_len:
            new_coords.append(coords[i])

    # return np.array(new_coords)
    return [x for x in new_coords]


    
def menu_extraction(fields, bboxes,texts, rel_s,column_coords_list):

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
