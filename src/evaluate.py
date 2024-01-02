import os, json
import re, cv2
import numpy as np
from tqdm import tqdm
from glob import glob

vis = False
# det_json_path = "result_rule_check/rico_test_dets.json"
# det_json_root = "result_rule_check_new"
det_json_root = "result_rule_check_FINAL_5Dec2022"
det_json_paths = glob(det_json_root +"/**.json")

print("hi", len(det_json_paths))
output_vis_folder = "result_rule_check_FINAL_5Dec2022"

# gt_json_path = "processed_data/Rico_testset/rico_test_annotations_6352.json"
gt_json_path = "processed_data/Rico_testset/rico_test_annotations6352_addPrivacy.json"


data_root = "processed_data/Rico_testset/testset"
all_imgs = os.listdir("processed_data/Rico_testset/testset")
print(len(all_imgs))


def is_matched(boxA, boxB, IOU_THRED=0):
    col_min_s = max(boxA[0], boxB[0])
    row_min_s = max(boxA[1], boxB[1])
    col_max_s = min(boxA[2], boxB[2])
    row_max_s = min(boxA[3], boxB[3])
    w = np.maximum(0, col_max_s - col_min_s)
    h = np.maximum(0, row_max_s - row_min_s)
    inter = w * h

    A_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    B_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter / (A_area + B_area - inter)

    if iou > IOU_THRED:
        return True
    return False

def draw_results(img, bbox_list, color):
    for bbox in bbox_list:
        img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), 
                                         color=color, thickness=5)

    # cv2.imshow("main", combined_img)
    # cv2.waitKey(0)

def evaluate_items(metrics, all_types, gt_items, det_items, img, img_id, output_vis_folder):
    TP = []
    FP = []
    FN = []
    results = {}

    for kk in all_types:
        # print("-->", kk)
        if vis:
            img_dp = img.copy()
        tmp_TP = []
        tmp_FP = []
        tmp_FN = []
        tmp_TP_gt = []
        inner_items = []
        if kk in ["SKIP", "NG-UPDATE", "II-AM-TQ", "FA-GAMIFICATION"]: #"II-AM-FH", 
            continue

        if kk not in metrics:
            metrics[kk] = {"TP":0, "FP":0, "FN":0}
        pred_item = det_items.get(kk, [])
        gt_item = gt_items.get(kk, [])

        flag_matched = [0]*len(pred_item)
        for gt in gt_item:
            flag_gt_match = False
            gt_bbox = gt["bbox"]

            for pred_idx, pred in enumerate(pred_item):
                pred_bbox = pred["bbox"]
                if flag_matched[pred_idx]:
                    continue

                # if gt and pred bbox is overlapped
                if is_matched(gt_bbox, pred_bbox):
                    flag_matched[pred_idx] = 1
                    tmp_TP.append(pred_bbox)
                    flag_gt_match = True

                    if kk in ["NG-AD", "II-AM-FH", "FA-G-COUNTDOWNAD"]:
                        for child in pred.get("children", []):
                            inner_items.append(child["bbox"])
                    break

            if not flag_gt_match:
                tmp_FN.append(gt_bbox)
            else:
                tmp_TP_gt.append(gt_bbox)

        for idx, flag in enumerate(flag_matched):
            if not flag:
                tmp_FP.append(pred_item[idx]["bbox"])

        metrics[kk]["TP"] += len(tmp_TP)
        metrics[kk]["FP"] += len(tmp_FP)
        metrics[kk]["FN"] += len(tmp_FN)

        if vis:
            draw_results(img_dp, tmp_TP_gt, color=(25,25,25))
            draw_results(img_dp, tmp_TP, color=(0,255,0))
            draw_results(img_dp, tmp_FP, color=(0,0,255))
            draw_results(img_dp, tmp_FN, color=(255,0,0))
            if len(inner_items) > 0:
                draw_results(img_dp, inner_items, color=(0,255,0))

        # print(tmp_FN, tmp_FP)
        # if len(tmp_FN) > 0 or len(tmp_FP) > 0:
        if vis:
            if len(tmp_FP) > 0:
                target_folder = os.path.join(output_vis_folder, kk, "FP")
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                target_path = os.path.join(target_folder, img_id)
                cv2.imwrite(target_path, img_dp)
                # cv2.imshow("main", img_dp)
                # cv2.waitKey(0)
            elif len(tmp_FN) > 0:
                target_folder = os.path.join(output_vis_folder, kk, "FN")
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                target_path = os.path.join(target_folder, img_id)
                cv2.imwrite(target_path, img_dp)
            elif len(tmp_TP) > 0:
                target_folder = os.path.join(output_vis_folder, kk, "TP")
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                target_path = os.path.join(target_folder, img_id)
                cv2.imwrite(target_path, img_dp)


        TP.extend(tmp_TP)
        FP.extend(tmp_FP)
        FN.extend(tmp_FN)
    return TP, FP, FN



def evaluate_screen(metrics, all_types, gt_items, det_items, img):
    for kk in all_types:
        if kk in ["SKIP", "NG-UPDATE", "II-AM-TQ"]: #"II-AM-FH", 
            continue

        if kk not in metrics:
            metrics[kk] = {"TP":0, "FP":0, "FN":0}
        pred_item = det_items.get(kk, [])
        gt_item = gt_items.get(kk, [])

        ### naive results
        pred_num = len(pred_item)
        gt_num = len(gt_item)

        if pred_num > gt_num:
            metrics[kk]["TP"] += gt_num
            metrics[kk]["FP"] += pred_num - gt_num
        elif pred_num < gt_num:
            metrics[kk]["TP"] += pred_num
            metrics[kk]["FN"] += gt_num - pred_num
        else:
            metrics[kk]["TP"] += pred_num
    return metrics


def calculate_metric(TP, FP, FN, name):
    if TP == 0 and FP == 0 and FN == 0:
        return
    precision = TP/(TP+FP) if (TP+FP) > 0 else 0
    recall = TP/(TP+FN) if (TP+FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    # print("==>", )
    print(f"{name:18s} GT: {TP+FN:3d} \tDET: {TP+FP:3d} & {precision:.02f} & {recall:.02f} & {f1:.02f}")
    return precision,recall,f1


def main(gt_json_path, det_json_path):
    curr_output_vis_folder = output_vis_folder + "_" + os.path.basename(det_json_path).split(".")[0]
    if vis:
        if not os.path.exists(curr_output_vis_folder):
            os.makedirs(curr_output_vis_folder)
    gt_data = json.load(open(gt_json_path, "r"))
    det_data = json.load(open(det_json_path, "r"))

    det_darkUI = [key for key, valus in det_data.items() if len(valus)!=0]
    gt_darkUI = [key for key, valus in gt_data.items() if len(valus)!=0 and "instances" in valus and len(valus["instances"])>0]
    print(f"\n\nDetect {len(det_darkUI)} dark UIs")
    print(f"GT {len(gt_darkUI)} dark UIs")
    FP_darkUI = set(det_darkUI) - set(gt_darkUI)
    print(f"FP Detect {len(FP_darkUI)} dark UIs")

    # with open("GT_darkUI.json", "w") as f:
    #     json.dump(gt_darkUI, f)
    # adss


    metrics = {}
    for img_id in tqdm(all_imgs):
        if ".jpg" not in img_id:
            continue

        # print("\n##", img_id)
        
        img_path = os.path.join(data_root, img_id)
        if vis:
            img = cv2.imread(img_path)
        else:
            img = None

        det_items = det_data.get(img_id, [])
        gt_items = gt_data.get(img_id, {})


        all_types = []
        if len(gt_items) != 0:
            gt_items = gt_items.get("instances", {})
            all_types.extend(list(gt_items.keys()))
        if len(det_items) != 0:
            all_types.extend(list(det_items.keys()))
        all_types = list(set(all_types))

        # if img_id == "67989.jpg":
        #     print("NG-AD" in gt_items)
        ## high-level
        # metrics = evaluate_screen(metrics, all_types, gt_items, det_items)

        # details
        TP, FP, FN = evaluate_items(metrics, all_types, gt_items, det_items, img, img_id, curr_output_vis_folder)

    # for kk in metrics.keys():
    #     precision,recall,f1 = calculate_metric(metrics["TP"], metrics["FP"], metrics["FN"], name)
    #     metrics[kk]["presision"] = precision
    #     metrics[kk]["recall"] = recall
    #     metrics[kk]["f1"] = f1

    ## higher level results
    print("==>", det_json_path)
    print("--> For each strategy")
    metric_cate = {}
    all_types = list(metrics.keys())
    all_types.sort()
    for kk in all_types:
        cate = kk.split("-")[0]
        # print(kk, cate)
        if cate not in metric_cate:
            metric_cate[cate] = {"TP":0, "FP":0, "FN":0}
        tmp_TP = metrics[kk]["TP"]
        tmp_FP = metrics[kk]["FP"]
        tmp_FN = metrics[kk]["FN"]
        metric_cate[cate]["TP"] += tmp_TP
        metric_cate[cate]["FP"] += tmp_FP
        metric_cate[cate]["FN"] += tmp_FN

    # for cate in sorted(metric_cate.keys()):
    for cate in ["NG", "SN", "II", "FA"]:
        precision, recall, f1 = calculate_metric(metric_cate[cate]["TP"], 
                  metric_cate[cate]["FP"], 
                  metric_cate[cate]["FN"], 
                  cate)
        metric_cate[cate]["precision"] = precision
        metric_cate[cate]["recall"] = recall
        metric_cate[cate]["f1"] = f1

    print("Avg precision/recall/f1:")
    print(round(sum(list(map(lambda x:x["precision"], metric_cate.values())))/len(metric_cate), 2), " & ", round(sum(list(map(lambda x:x["recall"], metric_cate.values())))/len(metric_cate), 2), " & ", round(sum(list(map(lambda x:x["f1"], metric_cate.values())))/len(metric_cate), 2))


    print("\n--> For each case")
    TP, FP, FN = 0,0,0
    for kk in all_types:
        tmp_TP = metrics[kk]["TP"]
        tmp_FP = metrics[kk]["FP"]
        tmp_FN = metrics[kk]["FN"]
        TP+=tmp_TP
        FP+=tmp_FP
        FN+=tmp_FN
        calculate_metric(tmp_TP, tmp_FP, tmp_FN, kk)

    print("\n--> Overall")
    calculate_metric(TP, FP, FN, "overall")


for det_json_path in det_json_paths:
    main(gt_json_path, det_json_path)




