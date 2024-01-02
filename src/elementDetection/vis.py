import json, cv2, os
from tqdm import tqdm
import numpy as np
import random
from pycocotools.coco import COCO
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [20, 12]
plt.rcParams["figure.autolayout"] = True


def cvt_bbox(bbox):
    bbox = [int(b) for b in bbox]
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

gt_json_filename = "GT/all_instances_test.json"
data_root = "../processed_data/Rico_testset/testset"

output_root = "vis_diff"
if not os.path.exists(output_root):
    os.makedirs(output_root)

dataset = "ricoOriText"
split = "test"
gt_dataset = dataset


gt_COCO = COCO(gt_json_filename)
all_imageids = gt_COCO.getImgIds()

# merge_paddle_results = json.load(open("examine_results/FRCNN_MERGED_PADDLE_COCOFM-evaluate.json", "r"))
merge_paddle_results = json.load(open("evaluate_results/new_merged_merge_ad_line-coco-nms0.7-evaluate.json", "r"))
oriText_frcnn_results = json.load(open("evaluate_results/FRCNN_ALL_nms0.7-evaluate.json", "r"))



def get_f1(data):
    GT_list = data["GT"]
    pred_list = data["predict"]

    leng = len(GT_list)
    print(leng)
    cp_GT_list = GT_list.copy()
    cp_pred_list = pred_list.copy()
    # for idx, item in enumerate(GT_list[::-1]):
    #     if item == 14:
    #         del cp_GT_list[leng-idx-1]
    #         del cp_pred_list[leng-idx-1]
    # leng = len(cp_pred_list)
    # for idx, item in enumerate(cp_pred_list.copy()[::-1]):
    #     if item == 14:
    #         del cp_GT_list[leng-idx-1]
    #         del cp_pred_list[leng-idx-1]

    print("+ GT", cp_GT_list)
    print("- PD", cp_pred_list)

    TP, FP, FN = 0,0,0
    for idx, gt in enumerate(cp_GT_list):
        pred = cp_pred_list[idx]
        if pred == gt:
            TP += 1
        elif pred is None:
            FN += 1
        elif gt is None:
            FP += 1
        elif pred != gt:
            FP += 1

    if TP == 0 and FP == 0 and FN == 0:
        return 0
    precision = TP/(TP+FP) if (TP+FP) > 0 else 0
    recall = TP/(TP+FN) if (TP+FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return f1


for img_id in tqdm(all_imageids):
    # if 2983 != img_id:
    #     continue
    img_info = gt_COCO.loadImgs(img_id)[0]
    size = [img_info["height"], img_info["width"]]
    annos = gt_COCO.getAnnIds(imgIds = img_id)


    img_path = os.path.join(data_root, str(img_id)+".jpg")
    print(img_path)
    print("--> Merged")
    merged_f1 = get_f1(merge_paddle_results[str(img_id)]["cate_details"])
    print("--> Ori")
    oritext_f1 = get_f1(oriText_frcnn_results[str(img_id)]["cate_details"])

    if oritext_f1 < merged_f1:
    # if True:
        print(oritext_f1, merged_f1)
        img = cv2.imread(img_path)
        size = [img_info["height"], img_info["width"]]

        gt_compos = []    
        gt_img = img.copy()
        for item_id in annos:
            item = gt_COCO.loadAnns(item_id)[0]
            cate = item["category_id"]
            bbox = cvt_bbox(item['bbox'])
            if cate == 14:
                gt_img = cv2.rectangle(gt_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (255,0,0), thickness = 3)
            else:
                print("GT", bbox)
                gt_img = cv2.rectangle(gt_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
        gt_img = cv2.putText(gt_img, "GT", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=3)

        img_ori = img.copy()
        for idx, bbox in enumerate(oriText_frcnn_results[str(img_id)]["cate_details"]["bbox"]):
            if bbox is None:
                continue
            cate = oriText_frcnn_results[str(img_id)]["cate_details"]["predict"][idx]
            if cate == 14:
                img_ori = cv2.rectangle(img_ori, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (255,0,0), thickness = 3)
                pass
            else:
                print("Ori", bbox)
                img_ori = cv2.rectangle(img_ori, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
        img_ori = cv2.putText(img_ori, "FRCNN_OriText", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=3)

        img_merge = img.copy()
        for idx, bbox in enumerate(merge_paddle_results[str(img_id)]["cate_details"]["bbox"]):
            if bbox is None:
                continue
            cate = merge_paddle_results[str(img_id)]["cate_details"]["predict"][idx]
            
            # if cate == 14:
            #     if bbox[1] < 0.025*size[0]  or bbox[3] > 0.95 * size[0]:
            #             continue
            if cate == 14:
                img_merge = cv2.rectangle(img_merge, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (255,0,0), thickness = 3)
                pass
            else:
                print("Merged", bbox)
                img_merge = cv2.rectangle(img_merge, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
        img_merge = cv2.putText(img_merge, "FRCNN_OriText+Paddle", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=3)
        blank = np.zeros((size[0], 10, 3), dtype= np.uint8)
        # plt.subplot(1, 3, 1), plt.imshow(gt_img, 'gray')
        # plt.subplot(1, 3, 2), plt.imshow(img_ori, 'gray')
        # plt.subplot(1, 3, 3), plt.imshow(img_merge, 'gray')
        target_path = os.path.join(output_root, str(img_id)+".jpg")
        im_v = cv2.hconcat([gt_img, blank, img_ori, blank, img_merge])
        cv2.imwrite(target_path, im_v)
        # plt.savefig(target_path) # To save figure
        # plt.show() # To show figure

        # cv2.imshow("asd", im_v)
        # cv2.waitKey()