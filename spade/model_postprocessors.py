import json
from thefuzz import fuzz
import re
from toolz import sliding_window


class FunctionNamespace(dict):
    def call_wrap(self, k):
        def wrapper(f):
            self[k] = f
        return wrapper

    def call_process(self, d):
        d = d.copy()
        for k, v in d.items():
            if k in self:
                d[k] = self[k](v)
        return d

    def __call__(self, arg):
        if isinstance(arg, str):
            return self.call_wrap(arg)
        else:
            return self.call_process(arg)


def find_max_idx(arr):
    max_score = max(arr)
    return arr.index(max_score)


class RemoveStopword:
    def __init__(self, stopwords, threshold=80):
        self.stopwords = [stopword.lower() for stopword in stopwords]
        self.threshold = threshold

    def get_score(self, field, stopword):
        if len(field) < len(stopword):
            return fuzz.token_set_ratio(field, stopword)
        n_window = len(stopword)
        windows = sliding_window(n_window, field)
        location = find_max_idx(
            [fuzz.ratio(''.join(w), stopword) for w in windows])
        weight = 1 - (location / len(field))
        return fuzz.token_set_ratio(field, stopword) * weight

    def __call__(self, field):
        stopwords = self.stopwords
        splits = re.split(r"[;:]", field)
        l_splits = [s.lower() for s in splits]
        idx = [i for (i, split) in enumerate(l_splits)
               if max([self.get_score(split, stopword)
                       for stopword in stopwords]) < self.threshold]
        splits = [splits[i] for i in idx]
        return ':'.join(splits).strip()


post_process_stopwords = FunctionNamespace()
# post_process_stopwords["info.date"] = RemoveStopword([])
post_process_stopwords['info.form'] = RemoveStopword(
    ["ký hiệu", "Ký hiệu (Serial No.)", "Ký hiệu (Serial)", "mẫu số (form)"])
post_process_stopwords["info.serial"] =\
    post_process_stopwords['info.form']
post_process_stopwords["info.num"] = RemoveStopword(
    ['số hiệu', "Số (Invoice No.)", "số hóa đơn", "số HĐ"])
post_process_stopwords["info.sign_date"] = RemoveStopword(
    ["ngày ký", "ký ngày"])
post_process_stopwords["seller.name"] = RemoveStopword(
    ["tên người bán", "đơn vị bán hàng"])
post_process_stopwords["seller.company"] = RemoveStopword(["đơn vị bán hàng"])
post_process_stopwords["seller.tax"] = RemoveStopword(
    ["mã số thuế", "MST", "tax code"])
post_process_stopwords["seller.tel"] = RemoveStopword(
    ["điện thoại", "số điện thoại", "tel", "số điện thoại (tel)"])
post_process_stopwords["seller.address"] = RemoveStopword(
    ["địa chỉ", "address", "địa chi", "dịa chỉ", "đia chi", "dia chi"])
post_process_stopwords["seller.bank"] = RemoveStopword(
    ["số tài khoản", "ngân hàng", "hình thức thanh toán", "Hình thức TT (Payment term)"])
post_process_stopwords["customer.name"] = RemoveStopword(
    ['người mua', 'đơn vị mua hàng', 'tên đơn vị'])
post_process_stopwords["customer.company"] = RemoveStopword(
    ['đơn vị mua hàng', 'tên đơn vị'])
post_process_stopwords["customer.tax"] =\
    post_process_stopwords["seller.tax"]
post_process_stopwords["customer.tel"] =\
    post_process_stopwords['seller.tel']
post_process_stopwords['customer.address'] =\
    post_process_stopwords['seller.address']
post_process_stopwords["customer.bank"] =\
    post_process_stopwords["seller.bank"]
post_process_stopwords["customer.payment_method"] = RemoveStopword(
    ['hình thức thanh toán'])
# post_process_stopwords["menu.product_id"] = RemoveStopword([])
# post_process_stopwords["menu.description"] = RemoveStopword([])
# post_process_stopwords["menu.unit"] = RemoveStopword([])
# post_process_stopwords["menu.quantity"] = RemoveStopword([])
# post_process_stopwords["menu.unit_price"] = RemoveStopword([])
# post_process_stopwords["menu.subtotal"] = RemoveStopword([])
# post_process_stopwords["menu.vat_rate"] = RemoveStopword([])
# post_process_stopwords["menu.vat"] = RemoveStopword([])
# post_process_stopwords["menu.total"] = RemoveStopword([])
post_process_stopwords["total.subtotal"] = RemoveStopword([
    'cộng tiền hàng'
])
post_process_stopwords["total.vat_rate"] = RemoveStopword(
    ['thuế suất', 'thuế suất GTGT'])
post_process_stopwords["total.vat"] = RemoveStopword(['GTGT'])
post_process_stopwords["total.total"] = RemoveStopword(
    ['tổng tiền thanh toán'])

stopwords = [
    "tên đơn vị bán hàng",
    "họ tên người mua hàng",
    "địa chỉ",
    "điện thoại",
    "phone",
    "email",
    "ngày lập",
    "ngày",
    "ngày ký",
    "ngày kí",
    "mã số thuế",
    "MST",
    "số tài khoản",
    "STK",
    "ngân hàng",
    "thuế suất GTGT",
    "Đơn vị bán",
    "người mua",
    "Cộng tiền hàng",
    "Hình thức thanh toán",
    "HTTT",
    "ký",
    "số",
    "(tax code)",
    "(address)",
    "mẫu số",
    "tên đơn vị (company)",
    "tel",
    "fax",
    "mẫu số (form)",
    "hàng (issued)",
    "email",
    "(payment method)",
    "ký hiệu (serial)",
    "ERC",
    "Tổng tiền thanh toán",
    "cộng tiền hàng",
    "address",
    "account no",
    "tiền thuế gtgt",
    "invoice no",
]


def remove_stopwords(s, stopwords=stopwords, threshold=80):
    scores = []
    slices = []
    for stopword in stopwords:
        idx, score = detect_stopword(s, stopword, threshold)
        scores.append(score + len(stopword))
        slices.append(idx)
    max_score = max(scores)
    idx = scores.index(max_score)
    s_idx = slices[idx]
    if s_idx is not None:
        s = s.replace(s[s_idx], '')
    s = s.strip(':; |`')
    return s


def remove_stopwords_2(result):
    for group_name, group in result.items():
        if isinstance(group, list):
            continue
        result[group_name] = post_process_stopwords(group)

    return result


def detect_stopword(s, stopword, threshold):
    n = len(stopword)
    windows = sliding_window(n, s)
    scores = []
    length = len(s)
    for i, window in enumerate(windows):
        segment = ''.join(window)
        score = fuzz.ratio(segment.lower(), stopword.lower())
        scores.append(score - i / length * 10)

    if len(scores) == 0:
        return None, 0
    max_score = max(scores)
    if max_score < threshold:
        return None, max_score
    else:
        start_index = scores.index(max_score)
        stop_index = start_index + n
        return slice(start_index, stop_index), max_score


def get_parsed_labeling(fields, rel_s):
    itc = rel_s[0:len(fields)]
    stc = rel_s[len(fields):]
    labeling = [-1]*len(rel_s[0])

    for i, _ in enumerate(stc):
        stc[i][i] = 0

    #head_box: danh sách box được label nối vào (box đầu của chuỗi box)
    head_box = []
    for i in range(len(fields)):
        for j in range(len(itc[i])):
            if itc[i][j] == 1 and labeling[j] == -1:
                head_box.append(j)
                labeling[j] = i


    bigbox_mapping = []
    while head_box != []:
        i = head_box[0]
        head_box.remove(i)
        new_box = [i]
        end_box = False
        while not end_box:
            end_box = True
            for j in range(len(stc[i])):
                if stc[i][j] == 1 and j not in new_box:
                    new_box.append(j)
                    i = j
                    end_box = False
                    break
        bigbox_mapping.append(new_box)
    return labeling, bigbox_mapping


def get_parsed_grouping_v2(fields, rel_g, labeling):
    itc = rel_g[0:len(fields)]
    stc = rel_g[len(fields):]
    grouping = [-1]*len(stc)
    head_group = []
    for i in range(len(stc)):
        for j in range(len(stc)):
            if stc[i][j] == 1 and i not in head_group:
                head_group.append(i)
    for k in range(len(head_group)):
        i = head_group[k]
        grouping[i] = k
        for j in range(len(stc)):
            if stc[i][j] == 1:
                grouping[j] = k

    return grouping


def get_parsed(fields, texts, labeling, grouping, bigbox_mapping):
    parsed = {}

    fields_rs = ["info", "seller", "customer", "total"]
    for i in fields_rs:
        parsed[f"{i}"] = {}
    parsed["menu"] = []
    parsed_menu = {}
    for i in grouping:
        if i != -1:
            parsed_menu[f"{i}"] = {}
    for bigbox in bigbox_mapping:
        head_id = bigbox[0]
        text = ""
        for i in bigbox:
            text += " " + texts[i]
        label = fields[labeling[head_id]]
        if grouping[head_id] != -1:
            parsed_menu[f"{grouping[head_id]}"][label] = text
        else:
            for field_rs in fields_rs:
                if field_rs in label:
                    parsed[f"{field_rs}"][label] = text

    group_key = [key for key in parsed_menu.keys()]
    for key in group_key:
        parsed["menu"].append(parsed_menu[key])
    return parsed

# Without rel-g


def get_parsed_grouping(fields, coords, labeling, bigbox_mapping):
    grouping = [-1]*len(coords)

    menudes_bigbox_mapping = []
    for bigbox in bigbox_mapping:
        head_id = bigbox[0]
        label = fields[labeling[head_id]]
        if label == "menu.description":
            menudes_bigbox_mapping.append(bigbox)

    all_rows = []
    for bigbox_des in menudes_bigbox_mapping:
        min_y = min([coords[box_id][0][1] for box_id in bigbox_des])
        max_y = max([coords[box_id][2][1] for box_id in bigbox_des])
        row = []
        for bigbox in bigbox_mapping:
            y_center = (coords[bigbox[0]][0][1]+coords[bigbox[0]][2][1])/2
            if y_center > min_y and y_center < max_y:
                row.append(bigbox)
        all_rows.append(row)

    for i in range(len(all_rows)):
        for bigbox in all_rows[i]:
            grouping[bigbox[0]] = i

    return grouping


def format_parsed_prettyly(parsed):
    fields_rs = [key for key in parsed.keys()]
    new_parsed = {}
    for field_rs in fields_rs:

        if field_rs == "menu":
            field_rs_dict_new = []
            for menu_dict in parsed[field_rs]:
                menu_dict_new = {}
                menu_dict_keys = [key for key in menu_dict.keys()]
                for key in menu_dict_keys:
                    if key.split('.')[0] != field_rs:
                        continue
                    menu_dict_new[key.replace(
                        f"{field_rs}.", "")] = menu_dict[key]
                field_rs_dict_new.append(menu_dict_new)
        else:
            field_rs_dict_new = {}
            field_rs_dict = parsed[field_rs]
            field_rs_dict_keys = [key for key in field_rs_dict.keys()]
            for key in field_rs_dict_keys:
                if key.split('.')[0] != field_rs:
                    continue
                field_rs_dict_new[key.replace(
                    f"{field_rs}.", "")] = field_rs_dict[key]
        new_parsed[field_rs] = field_rs_dict_new

    return new_parsed


def parsed_menu_update(parsed):
    menu_list = parsed["menu"]
    main_key = ["quantity", "total", "unit_price"]
    # sub_key=["subtotal","vat_rate"]
    new_menu = []
    for menu in menu_list:
        menu_keys = [key for key in menu.keys()]
        missed_key = []
        for key in main_key:
            if key not in menu_keys:
                missed_key.append(key)
        if len(missed_key) > 1 or missed_key == []:
            try:
                #Trường hợp số lượng = 0 do ocr sai
                
                if eval(menu["quantity"]) == 0  and eval(menu["unit_price"])!=0:
                    print("got one")
                    menu["quantity"] = eval(menu["total"])/eval(menu["unit_price"])

            except:
                pass
            new_menu.append(menu)
            continue

        for key in main_key:
            try:
                menu[key] = menu[key].replace(",", "")
            except:
                pass
        
        #Trường hợp bị thiếu label
        try:
            if missed_key[0] == "quantity":
                menu["quantity"] = eval(menu["total"])/eval(menu["unit_price"])
            if missed_key[0] == "total":
                menu["total"] = eval(menu["quantity"])*eval(menu["unit_price"])
            if missed_key[0] == "unit_price":
                menu["unit_price"] = eval(menu["total"])/eval(menu["quantity"])
        except:
            pass
        
        new_menu.append(menu)

    parsed["menu"] = new_menu
    return parsed


if __name__ == "__main__":
    with open("dataset/invoice_ie/test_invoice_vn.jsonl") as f:
        data = json.loads(f.readline())

    labeling, bigbox_mapping = get_parsed_labeling(
        data["fields"], data["label"][0])
    # grouping=get_parsed_grouping(data["fields"],data["label"][1],labeling)

    grouping = get_parsed_grouping(
        data["fields"], data["coord"], labeling, bigbox_mapping)
    parsed = get_parsed(data["fields"], data["text"],
                        labeling, grouping, bigbox_mapping)
    parsed = format_parsed_prettyly(parsed)
    parsed = parsed_menu_update(parsed)
