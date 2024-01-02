# faster rcnn
import os, sys, json, cv2
# sys.path.append("/media/cheer/UI/Project/DarkPattern/code/detectionFromPixel/Detection_Models/FASTER_RCNN/lib")
# from model.roi_layers import nms
from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [20, 12]
plt.rcParams["figure.autolayout"] = True

from wordsegment import load, segment
load()

### BE CAREFUL. faster rcnn may fail to detect items in some UIs, this will lead to the results that our merging output will not contain them!!!
## change to ask to provide a list of target img!!


def draw_bbox(after_nms):
    for input_file, result_list in after_nms.items():
        # print(input_file)
        img = cv2.imread(input_file)
        for item in result_list:
            bbox = item["bbox"]
            class_name = item["category"]
            score = item["score"]
            # print(bbox)
            cv2.rectangle(img, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(img, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
        result_path = input_file.replace(".jpg", "-nms-addText.jpg")
        cv2.imwrite(result_path, img)


all_types = ['Button', 'CheckBox', 'Chronometer', 'EditText', 
             'ImageButton', 'ImageView',
            'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
            'ToggleButton', 'VideoView', 'TextView']
cate2int = {cate:idx for idx, cate in enumerate(all_types)}
cate2int["pText"] = cate2int["TextView"]


nms_threshold = 0.1

def merge_dets_single(fasterRCNN_dets, OCR_dets, img_cv, output_json, vis=False):
    merged_results = []
    text_items = OCR_dets
    all_items = fasterRCNN_dets
    if vis:
        img = img_cv.copy()
    # try:
    if True:
        if len(all_items) == 0:
            for text_item in text_items:
                text_item["category"] = "pText"
                if text_item["text"] == "口":
                        continue
                if text_item["text"] == "三":
                    text_item["category"] = "ImageButton"
                merged_results.append(text_item)

        if vis:
            #-  --------- DRAW RAW DETS
            img_raw = img.copy()
            for item in all_items:
                bbox = item["bbox"]
                cate = item["category"]
                # if item["score"] < 0.65:
                #     continue
                if cate == "TextView":
                    img_raw = cv2.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,0,255), thickness = 5)
                else:
                    img_raw = cv2.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
            #- DRAW RAW DETS --------- 

        all_items.sort(key=lambda x:x["score"], reverse=True)
        pred_boxes = list(map(lambda x:x["bbox"], all_items))
        cls_scores = list(map(lambda x:x["score"], all_items))
        category = list(map(lambda x:x["category"], all_items))


        tmp_valid_item = []
        overlay_threshold = 0.5
        ## iterate each detection and remove duplicates
        for item in all_items:
            tmp_box = item["bbox"]
            flag = True
            for valid_item in tmp_valid_item:
                valid_item_bbox = valid_item["bbox"]

                # in valid_item
                ## if existing selection contains current detection
                if tmp_box[0] >= valid_item_bbox[0] and tmp_box[1] >= valid_item_bbox[1] \
                    and tmp_box[2] <= valid_item_bbox[2] and tmp_box[3] <= valid_item_bbox[3]:
                    # directly skip it
                    ## e.g. text element(current) in the button(existing), we can skip the text element
                    flag = False
                    break

                # contain valid item
                ## if current detection is contained by existing selection
                if tmp_box[0] <= valid_item_bbox[0] and tmp_box[1] <= valid_item_bbox[1] \
                    and tmp_box[2] >= valid_item_bbox[2] and tmp_box[3] >= valid_item_bbox[3]:
                    ## e.g. text element(existing) in the button(current), we should keep the button
                    # if valid_item["category"] in ["TextView"] and item["category"] == "Button":
                    if valid_item["category"] in ["TextView", "ImageView"] and item["category"] == "Button":
                        continue

                    flag = False
                    break

            if flag:
                tmp_valid_item.append(item)

        # print("len(tmp_valid_item)", len(tmp_valid_item))
        tmp_valid_item.sort(key=lambda x:(x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True)

        ## remove texts/images that inside a button
        tmp_valid_item_update = []
        for item in tmp_valid_item:
            tmp_box = item["bbox"]
            flag = True
            for valid_item in tmp_valid_item_update:
                valid_item_bbox = valid_item["bbox"]

                # in valid_item
                if tmp_box[0] >= valid_item_bbox[0] and tmp_box[1] >= valid_item_bbox[1] \
                    and tmp_box[2] <= valid_item_bbox[2] and tmp_box[3] <= valid_item_bbox[3]:
                    flag = False
                    break

            if flag:
                tmp_valid_item_update.append(item)
                
        # print("len(tmp_valid_item_update):", len(tmp_valid_item_update))
        # # get paddle results
        # if imgid not in all_ocr_data:
        #     merged_results = all_items
        #     # continue
        #     continue


        if vis:
            #-  --------- DRAW TEXT DETS
            img_text = img.copy()
            for item in text_items:
                bbox = item["bbox"]
                img_text = cv2.rectangle(img_text, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,0,255), thickness = 5)
            #- DRAW TEXT DETS --------- 



        # sort by area
        tmp_valid_item_update.sort(key=lambda x:(x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=False)
        # merge with nontext results
        final_items = []
        # print("len(text_items):", len(text_items))
        flag_system_time_text = False
        for text_item in text_items:
            text_bbox = text_item["bbox"]
            text_score = text_item["score"]
            text = text_item["text"]

            if ":" in text:
                if text_bbox[3] <= 50 or text_bbox[1]<=50:
                    # print("skip time", text_item["text"])
                    flag_system_time_text = True
                    continue
            if flag_system_time_text:
                if text_bbox[3] <= 50 or text_bbox[1]<=50:
                    # print("skip time", text_item["text"])
                    continue 

            # print("----", text)

            best_match = None
            best_iou = 0
            for valid_item in tmp_valid_item_update:
                valid_item_bbox = valid_item["bbox"]

                # if text box does not overlap with item box
                if (max(text_bbox[2], valid_item_bbox[2]) - min(text_bbox[0], valid_item_bbox[0]))>= (text_bbox[2] + valid_item_bbox[2] - text_bbox[0] -valid_item_bbox[0]) and \
                    (max(text_bbox[3], valid_item_bbox[3]) - min(text_bbox[1], valid_item_bbox[1]))>= (text_bbox[3] + valid_item_bbox[3] - text_bbox[1] -valid_item_bbox[1]):
                    iou = 0
                else:
                    # else, calculate iou
                    x1,y1,x2,y2 = max(text_bbox[0], valid_item_bbox[0]), \
                                  max(text_bbox[1], valid_item_bbox[1]), \
                                  min(text_bbox[2], valid_item_bbox[2]), \
                                  min(text_bbox[3], valid_item_bbox[3])

                    inter_area = (y2-y1) * (x2-x1)
                    text_ares = (text_bbox[2]-text_bbox[0])*(text_bbox[3]-text_bbox[1])
                    valid_area = (valid_item_bbox[2]-valid_item_bbox[0])*(valid_item_bbox[3]-valid_item_bbox[1])
                    iou = inter_area/(text_ares+valid_area-inter_area)

                if iou == 0:
                    continue
                if text_bbox[0] >= valid_item_bbox[0] and text_bbox[1] >= valid_item_bbox[1] \
                    and text_bbox[2] <= valid_item_bbox[2] and text_bbox[3] <= valid_item_bbox[3]:
                    tmp_match = valid_item
                    tmp_iou = 1 + iou

                    # print("!")
                    if tmp_iou > best_iou:
                        best_iou = tmp_iou
                        best_match = valid_item
                elif iou > 0:
                    if iou > best_iou:
                        best_iou = iou
                        best_match = valid_item

            if best_match is None:
                # print("text_bbox", text_bbox)
                if text_item["text"] == "口":
                    continue
                if text_item["text"] == "三":
                    text_item["category"] = "ImageButton"
                final_items.append(text_item)
                # print("- no matched", text_item)
            else:

            # if best_match:
                # print("+ matched", best_match["category"])
                best_match["matched"] = True
                if "text_items" not in best_match:
                    best_match["text_items"] = []
                best_match["text_items"].append(text_item)
                # print(best_match["score"], best_match["bbox"], text_item["score"], text_item["text"], text_item)
                best_match["score"] = max(best_match["score"], text_item["score"])

                # merge_bbox = [min(text_bbox[0], best_bbox[0]), \
                #               min(text_bbox[1], best_bbox[1]), \
                #               max(text_bbox[2], best_bbox[2]), \
                #               max(text_bbox[3], best_bbox[3])]
                # best_match["bbox"] = merge_bbox
                # # print("merged:", merge_bbox)
                final_items.append(best_match)
                best_match = None
                best_iou = 0

        # for remaining item box that does not match with any text items, we directly add them into final list
        for item in tmp_valid_item_update:
            if "matched" not in item:
                # we believe paddle can must detect the text if it exists. 
                # Therefore, if the item does not match with the text item, it may be wrong detections
                #  One problem worths to mention is that some icons are considered as a textview!!!!!!  @@@@
                if item["category"] != "TextView":
                    final_items.append(item)
                # print("lost:", item["bbox"])


        if vis:
            #-  --------- DRAW FINAL DETS
            img_final = img.copy()
            for item in final_items:
                bbox = item["bbox"]
                cate = item["category"]
                if item["score"] < 0.7 and item["category"] == "pText":
                    # print("deleted:", item.get("text_items", [{"text":""}])[0]["text"])
                    continue
                if cate == "TextView":
                    img_final = cv2.rectangle(img_final, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,0,255), thickness = 5)
                else:
                    img_final = cv2.rectangle(img_final, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
            #- DRAW FINAL DETS --------- 

        for item in final_items:
            if "matched" in item:
                tmp_texts_items = item["text_items"]
                item_bbox = item["bbox"]
                if len(tmp_texts_items) == 1:
                    text_item = tmp_texts_items[0]
                    text_bbox = text_item["bbox"]
                    text = text_item["text"]
                else:
                    # contain multiple texts
                    tmp_texts_items.sort(key=lambda x:(x["bbox"][1],x["bbox"][0]), reverse=False)
                    texts = list(map(lambda x:x["text"], tmp_texts_items))
                    # print(tmp_texts_items)
                    text_x1 = min([int(a["bbox"][0]) for a in tmp_texts_items])
                    text_y1 = min([int(a["bbox"][1]) for a in tmp_texts_items])
                    text_x2 = max([int(a["bbox"][2]) for a in tmp_texts_items])
                    text_y2 = max([int(a["bbox"][3]) for a in tmp_texts_items])

                    text_bbox = [text_x1, text_y1, text_x2, text_y2]
                    text = " ".join(texts)

                    
                item["ori_text"] = text
                try:
                    text_tokens = segment(text)
                    text = " ".join(text_tokens)
                except:
                    pass
                # print("++ ", text)
                # print(item["category"])
                if item["category"] in ["TextView", 14]:
                    # item["bbox"] = text_bbox
                    ## ------------------- 11 Nov 2022  --------
                    # check iou first? some textview may be text button!
                    x1,y1,x2,y2 = max(text_bbox[0], item_bbox[0]), \
                                  max(text_bbox[1], item_bbox[1]), \
                                  min(text_bbox[2], item_bbox[2]), \
                                  min(text_bbox[3], item_bbox[3])

                    inter_area = (y2-y1) * (x2-x1)
                    text_ares = (text_bbox[2]-text_bbox[0])*(text_bbox[3]-text_bbox[1])
                    valid_area = (item_bbox[2]-item_bbox[0])*(item_bbox[3]-item_bbox[1])
                    tmp_text_iou = inter_area/(text_ares+valid_area-inter_area)
                    if tmp_text_iou > 0.5:
                        item["bbox"] = text_bbox
                    else:
                        item["bbox"] = item_bbox
                    # print("hello")
                    # ------------------- 11 Nov 2022  --------  ##
                else:
                    merge_bbox = [min(text_bbox[0], item_bbox[0]), \
                                  min(text_bbox[1], item_bbox[1]), \
                                  max(text_bbox[2], item_bbox[2]), \
                                  max(text_bbox[3], item_bbox[3])]
                    item["bbox"] = merge_bbox
                item["text"] = text

        ## merge same height texts
        final_final_items = []
        all_pText = []
        for item in final_items:
            if item["category"] != "pText":
                final_final_items.append(item)
            else:
                all_pText.append(item)

        all_pText = sorted(all_pText, key = lambda x:(x["bbox"][1], x["bbox"][0]))
        final_pText = []
        tmp_text = []
        flag_yes = False
        for pText in all_pText:
            if len(final_pText) == 0:
                final_pText.append([pText])
                # print("First!", pText)
            else:
                # print(pText["bbox"], pText["text"])
                for idx, ptexts in enumerate(final_pText):
                    x1_a, y1_a, x2_a, y2_a = ptexts[-1]["bbox"]
                    w_a, h_a = x2_a - x1_a, y2_a-y1_a
                    xc_a = (x1_a+x2_a)//2
                    x1_b, y1_b, x2_b, y2_b = pText["bbox"]
                    w_b, h_b = x2_b - x1_b, y2_b-y1_b
                    xc_b = (x1_b+x2_b)//2
                    # print("++", pText["text"], pText["bbox"], "-", ptexts[-1]["text"], "-", ptexts[-1]["bbox"], "-", )
                    # print(abs(h_a - h_b), min(abs(y1_b - y2_a), abs(y1_a - y2_b)), min(h_a,h_b)//2 , w_a+w_b, min(x2_a, x2_b) - max(x1_a, x1_b))

                    if abs(h_a - h_b) < 10 and min(abs(y1_b - y2_a), abs(y1_a - y2_b)) < min(h_a,h_b)//2 and ((min(x2_a, x2_b) - max(x1_a, x1_b))/(max(x2_a, x2_b) - min(x1_a, x1_b)) > 0.5 or abs(xc_a-xc_b)<10):
                        final_pText[idx].append(pText)
                        # print("a[[]]")
                        flag_yes = True
                        break
                    # print(flag_yes)
                if not flag_yes:
                    final_pText.append([pText])
                    # print("Add new text", pText["text"])
                flag_yes = False

        for ptexts in final_pText:
            if len(ptexts) == 1:
                final_final_items.append(ptexts[0])
            else:
                ptexts.sort(key=lambda x:(x["bbox"][1],x["bbox"][0]), reverse=False)
                texts = list(map(lambda x:x["text"], ptexts))
                text = " ".join(texts)
                # print(ptexts)
                text_x1 = min([int(a["bbox"][0]) for a in ptexts])
                text_y1 = min([int(a["bbox"][1]) for a in ptexts])
                text_x2 = max([int(a["bbox"][2]) for a in ptexts])
                text_y2 = max([int(a["bbox"][3]) for a in ptexts])

                score = max([a["score"] for a in ptexts])

                text_bbox = [text_x1, text_y1, text_x2, text_y2]
                final_final_items.append({"category":"pText",
                                          "text": text,
                                          "bbox": text_bbox,
                                          "score":score})

        final_items = final_final_items
        merged_results = final_items

        if vis:
            #-  --------- DRAW FINAL DETS
            img_final2 = img.copy()
            for item in final_items:
                bbox = item["bbox"]
                cate = item["category"]
                # if item["score"] < 0.7:
                if item["score"] < 0.7 and item["category"] == "pText":
                    continue
                if item["score"] < 0.65:
                    continue
                if cate == "TextView":
                    img_final2 = cv2.rectangle(img_final2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,0,255), thickness = 5)
                else:
                    img_final2 = cv2.rectangle(img_final2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = (0,255,0), thickness = 3)
            #- DRAW FINAL DETS --------- 

            plt.subplot(1, 4, 1), plt.imshow(img_raw, 'gray')
            plt.subplot(1, 4, 2), plt.imshow(img_text, 'gray')
            plt.subplot(1, 4, 3), plt.imshow(img_final, 'gray')
            plt.subplot(1, 4, 4), plt.imshow(img_final2, 'gray')
            plt.savefig(output_json.replace(".json", ".jpg")) # To save figure
            plt.show() # To show figure

        # except:
        #     pass
            # print("ERROR", imgid)
    # except:
    #     # continue
        # pass
    with open(output_json, "w") as f:
        json.dump(merged_results, f)
    return merged_results

