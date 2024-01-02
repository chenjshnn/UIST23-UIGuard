import os
import json
import cv2, math
import random
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

classSet = ['TextView', 'ImageView', 'RadioButton', 'pText', 'ProgressBar', 'CheckBox', 'ToggleButton', 'EditText', 'ImageButton', 'SeekBar', 'Switch', 'RatingBar', 'Button', "Other"]
class2id = {name:idx for idx, name in enumerate(classSet)}
id2class = {v:k for k,v in class2id.items()}

def extract_feats(meta, img_h, img_w):
    ''' extract feats from json for clustering '''
    leaf_node, leaf_node_text = [], []
    for item_idx, item in enumerate(meta):
        item_bbox = item["bbox"]
        item_class = item["category"]
        ## pText is the text elment detected by OCR 
        if item_class == "ptext":
            item_class = "TextView"
        item_text = item.get("text","")

        ## feat [normalized_w, normalized_h, classId, num_words, num_node, *curr_ele_bbox]
        feat = [(item_bbox[2]-item_bbox[0])/img_w, 
                (item_bbox[3]-item_bbox[1])/img_h, 
                class2id.get(item_class, class2id["Other"]), 
                len(item_text.split(" ")), 
                item_idx,
                *item_bbox]

        leaf_node.append(feat)
        leaf_node_text.append([item_text, item_class])
    return leaf_node, leaf_node_text


def custom_metric(featA, featB):
    ''' DBSCAN distance function '''
    ## feat [normalized_w, normalized_h, classId, len_text, num_node, *curr_ele_bbox]

    sim = 0
    ## normalized_w
    # if featA[0] == featB[0]: 
    # sim += 2*min(featA[0], featB[0])/max(featA[0], featB[0])
    sim += 2*(1-abs(featA[0] - featB[0])/max(featA[0], featB[0]))

    ## normalized_h
    # if featA[1] == featB[1]:
    sim += 2*(1-abs(featA[1]-featB[1])/max(featA[1], featB[1]))

    ## classId
    if featA[2] == featB[2]: 
        sim += 2

    ## len_text
    # if not (featA[3]==0 and featB[3] == 0):
    #     sim += min(featA[3], featB[3]) / max(featA[3], featB[3])
    # if featA[3] < 15 and featB[3] < 15:
    #     sim += 3

    ## semantic relationship: (bert embedding)
    # if featA[4] != -10 and featB[4] != -10:
    #     cosine_sim = cosine(featA[4:4+768], featB[4:4+768])
    #     sim += 5*cosine_sim

    ## centroid dist
    x1_a, y1_a, x2_a, y2_a = featA[-4:]
    x1_b, y1_b, x2_b, y2_b = featB[-4:]

    f1x = (x1_a+x2_a)/ 2
    f1y = (y1_a+y2_a)/ 2

    f2x = (x1_b+x2_b)/ 2
    f2y = (y1_b+y2_b)/ 2

    if abs(f1y-f2y) < 50  or abs(f1x-f2x) < 50:
        sim += 0.5
    # print("#1", sim)

    # if abs(f1x-f2x) < 50:# or abs(f1x-f2x) < 1080//3:
    #     sim += 0.5

    # distance
    if abs(f1y-f2y) < 50:
        hori_dist = min(abs(x1_a-x2_b), abs(x2_a-x1_b))
        if hori_dist <= 50:
            sim+=0.5
        else:
            sim-=0.5
            # print("#2", sim)
    # elif abs(f1x-f2x) < 50:
    #     vert_dist = min(abs(y1_a-y2_b), abs(y2_a-y1_b))
    #     if vert_dist <= 150:
    #         sim+=1
    #         print("#3", sim)

    sim /= 6
    sim = max(0, sim)
    # print("Diff {} - {}: {}".format(featA[-5], featB[-5], 1-sim))

    return 1-sim

def get_text_button(meta):
    ''' extract feats from json for clustering '''
    new_meta = []
    for item_idx, item in enumerate(meta):
        item_bbox = item["bbox"]
        item_class = item["category"]
        print(item_class)
        if item_class not in ["TextView", "pText", "Button"]:
            continue

        if item_bbox[1] <50:
            continue
        ## pText is the text elment detected by OCR 
        if item_class == "ptext":
            item_class = "TextView"
            item["category"] = "TextView"
        item["ID"] = int(item_idx)
        item_text = item.get("text","")
        if len(item_text) == 0:
            continue
        if len(item_text.split()) > 5:
            continue
        new_meta.append(item)
    return new_meta


def get_grouping(meta, h, w):
    sensitive_words = ["close", "not now", "already have", 
                       "next time", "close", "no thanks", 
                       "later", "skip", "cancel", 
                       "sign", "register", "I'll", "I will", 
                       "I have", "I've"]

    new_meta = get_text_button(meta)
    leaf_node, leaf_node_text = extract_feats(new_meta, h, w)
    if len(leaf_node) <= 1:
        return {}
    clustering = DBSCAN(eps=0.3, min_samples = 1, metric=custom_metric).fit(leaf_node)
    # print("clustering labels:", clustering.labels_)#, clustering._dist)
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    gid2idx = {}
    unique = unique.astype(np.int32).tolist()
    for gid in unique:
        eleids = np.where(clustering.labels_ == gid)[0]
        gid2idx[gid] = eleids.tolist()
        # print(type(gid2idx[gid][0]) == type(1), type(gid))
    # print(gid2idx)

    ### 
    # needToIncludeIdx = []
    # for idx, item in enumerate(meta):
    #     item_text = item["text"]
    #     flag_contains = list(map(lambda x: x in item_text, sensitive_words))
    #     if len(item_text.split()) < 15 and sum(flag_contains) > 0:
    #         needToIncludeIdx.append(idx)

    # apply some rules
    cleaned_idx = [-1] * len(meta)

    final_gid2idxes = {}
    for gid, idxes in gid2idx.items():
        # flag_below = False
        # flag_top = False
        elecons_bg = []
        elecons_fg = []
        ele_texts = []
        elecates = set()
        for idx in idxes:
            item = new_meta[idx]
            bbox = item["bbox"]
            text = item.get("text", "").lower()
            cate = item["category"]
            bg_lum = item["bg_lum"]
            fg_lum = item["fg_lum"]

            elecates.add(cate)
            ele_texts.append(text)
            if bg_lum is not None:
                elecons_bg.append(bg_lum)
            if fg_lum is not None:
                elecons_fg.append(fg_lum)

        elecates = list(elecates)
        # print(elecates)
        # final_gid2idxes[gid] = []
        # for idx in idxes:
        #     item = new_meta[idx]
        #     tmp_id = item["ID"]
        #     final_gid2idxes[gid].append(tmp_id)
        if 1< len(idxes)<8 and "Button" in elecates: 
            flag_dark = False
            if (max(elecons_bg) - min(elecons_bg)) > 10:
                flag_dark = True
            elif (max(elecons_fg) - min(elecons_fg)) > 10:
                flag_dark = True
            if flag_dark:
                # check text content
                # flag = False
                # for ele_text in ele_texts:
                #     flag_contains = list(map(lambda x: x in ele_text, sensitive_words))
                #     if sum(flag_contains) > 0 or "no" == ele_text:
                #         flag = True
                #         print("contained")
                #         # print(sensitive_words[flag_contains.index(True)])
                #         break
                # if flag:
                    final_gid2idxes[gid] = []
                    for idx in idxes:
                        item = new_meta[idx]
                        tmp_id = item["ID"]
                        final_gid2idxes[gid].append(tmp_id)
                        cleaned_idx[tmp_id] = gid
                        # print(item.get("text", ""))
                        # if tmp_id in needToIncludeIdx:
                        #     needToIncludeIdx.remove(tmp_id)

    return final_gid2idxes

import cv2
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.get_cmap("tab10")


cv2.namedWindow("main", cv2.WINDOW_NORMAL)
def draw_results(img_path, output_results, meta):
    img = cv2.imread(img_path)

    for gid, idxes in output_results.items():
        if gid == -1:
            curr_color = (222,222,222)
        else:
            curr_color = cmap(gid % 10)
            curr_color = [int(255*a) for a in curr_color][:3]

        for idx in idxes:
            item = meta[idx]
    # for idx, item in enumerate(meta):
    #     gid = output_results[idx]
    #     # if gid == -1:
    #     #     continue
    #     # if item["category"] not in ["TextView", "Button"]:
    #     #     continue
        
            x1,y1,x2,y2 = item["bbox"]
            img = cv2.rectangle(img, (x1,y1), (x2,y2), 
                                         color=curr_color, thickness=5)
    cv2.imshow("main", img)
    cv2.waitKey(0)
    return img


if __name__ == '__main__':
    
    from glob import glob
    data_root = "../processed_data/Rico_testset/testset"

    # target_img = os.listdir("../processed_data/Rico_testset/vis/AM-FH")
    target_img = os.listdir("../processed_data/Rico_testset/testset")

    target_img.sort()

    json_root = "../new_gather_results_addTM"
    # json_root = "../gather_info_frcnn_addTM"

    # all_jsons = glob(json_root+"/**.json")
    # all_jsons.sort()

    saved_root = "grouping_2"
    if not os.path.exists(saved_root):
        os.makedirs(saved_root)

    all_dets = {}
    for img_id in tqdm(target_img):
        if img_id not in ["41691.jpg"]: # "2767.jpg", 
            continue
        if img_id.split(".")[-1].lower() not in ["png", "jpg","jpeg"]: 
            continue
        print(img_id)

        # img_id = os.path.basename(json_path).replace(".json", ".jpg")
        img_path = os.path.join(data_root, img_id)

        json_path = os.path.join(json_root, img_id.replace(".jpg", ".json"))
        json_path = json_path.replace(".png", ".json")
        # print(json_path)

        if not os.path.exists(json_path):
            print("Failed to find metadata", img_id)
            continue

        meta = json.load(open(json_path, "r"))

        img = cv2.imread(img_path)
        h,w,_ = img.shape
        # print(img_path)

        # meta = json.load(open(json_path, "r"))
        final_gid2idxes = get_grouping(meta, h,w)

        # saved_path = os.path.join(saved_root, os.path.basename(json_path))
        # json.dump(output_results, json.open(saved_path, "w"))
        img = draw_results(img_path, final_gid2idxes, meta)
        saved_path = os.path.join(saved_root, img_id)
        # cv2.imwrite(saved_path,img)


        # target_json = os.path.join(saved_root, os.path.basename(json_path))
        # print("###", final_gid2idxes)
        # with open(target_json, "w") as f:
        #     json.dump(final_gid2idxes, f)

 