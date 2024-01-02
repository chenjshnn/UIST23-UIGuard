import os, json
import re
from tqdm import tqdm

def merge_bbox(items):
    all_bbox = [item["bbox"] for item in items]
    x1 = min([bbox[0] for bbox in all_bbox])
    y1 = min([bbox[1] for bbox in all_bbox])
    x2 = max([bbox[2] for bbox in all_bbox])
    y2 = max([bbox[3] for bbox in all_bbox])
    return [x1,y1,x2,y2] 

def is_horizontal_overlap(itemA_bbox, itemB_bbox):
    A_h = itemA_bbox[3] - itemA_bbox[1]
    B_h = itemB_bbox[3] - itemB_bbox[1]

    tmp_dist = max(itemA_bbox[3], itemB_bbox[3]) - min(itemA_bbox[1], itemB_bbox[1])

    if tmp_dist < A_h + B_h:
        return True
    else:
        return False
        
def merge_checkbox_w_text(item_list):
    all_checkboxes = [item for item in item_list if item["category"] in ["CheckBox", "ToggleButton", "Switch"]]

    check_icons = [item for item in item_list if item.get("iconLabel", [None])[0] == "check"]
    for check_icon in check_icons:
        check_icon["status"] = ["checked", 1]

    all_checkboxes.extend(check_icons)

    flag = [0] * len(item_list)
    for check_idx in range(len(all_checkboxes)):
        checkbox = all_checkboxes[check_idx]
        checkbox_bbox = checkbox["bbox"]
        checkbox_h = checkbox_bbox[3] - checkbox_bbox[1]
        checkbox["match_text"] = []
        for item_idx in range(len(item_list)):
            item = item_list[item_idx]
            item["id"] = item_idx
            cate = item["category"]
            if cate not in ["TextView", "pText"]:
                continue
            if flag[item_idx]:
                continue

            item_bbox = item["bbox"]
            if is_horizontal_overlap(checkbox_bbox, item_bbox):
                flag[item_idx] = 1
                checkbox["match_text"].append(item)

    def check_horizontal_overlap(text_items, checkbox_bbox):
        removed_idx = []
        for idxA, itemA in enumerate(text_items):
            if idxA in removed_idx:
                continue
            itemA_bbox = itemA["bbox"] 

            for idxB, itemB in enumerate(text_items):
                if idxB<=idxA:
                    continue
                if idxB in removed_idx:
                    continue
                itemB_bbox = itemB["bbox"]
                if is_horizontal_overlap(itemA_bbox, itemB_bbox):
                    distA = min(abs(checkbox_bbox[0]-itemA_bbox[2]), abs(checkbox_bbox[2]-itemA_bbox[0]))
                    distB = min(abs(checkbox_bbox[0]-itemB_bbox[2]), abs(checkbox_bbox[2]-itemB_bbox[0]))
                    if distA < distB:
                        removed_idx.append(idxB)
                    elif distA > distB:
                        removed_idx.append(idxA)
        return removed_idx


    final_checkbox_group = []
    skip_idx = []
    single_checkbox = []
    for check_idx in range(len(all_checkboxes)):
        flag_match = False
        checkbox = all_checkboxes[check_idx]
        checkbox_bbox = checkbox["bbox"]
        matched_texts = checkbox["match_text"] 
        if len(matched_texts) == 0:
            # print("missing text")
            pass
        if len(matched_texts) == 1:
            merged_bbox = merge_bbox([checkbox, matched_texts[0]])
            if checkbox.get("text","").strip().lower() == "on":
                status = ["checked", 1]
            else:
                status = checkbox["status"]
            tmp = {"category": "check_group", 
                   "bbox":merged_bbox, 
                   "text": matched_texts[0]["text"], 
                   "status": status,
                   "meta_items": [checkbox, matched_texts[0]]}
            final_checkbox_group.append(tmp)
            flag_match = True

            skip_idx.append(checkbox["id"])
            for text_item in matched_texts:
                skip_idx.append(text_item["id"])

        else:
            removed_idx = check_horizontal_overlap(matched_texts, checkbox_bbox)
            # removed_idx.sort(r)
            corrected_text_items = [text_item for text_idx, text_item in enumerate(matched_texts) if text_idx not in removed_idx] 
            if len(corrected_text_items) == 0:
                # print("missing text")
                pass
            else:
                merged_bbox = merge_bbox([*corrected_text_items,checkbox])
                if checkbox.get("text","").strip().lower() == "on":
                    status = ["checked", 1]
                else:
                    status = checkbox["status"]

                tmp = {"category": "check_group", 
                       "bbox":merged_bbox, 
                       "text": ",".join([text_item["text"] for text_item in corrected_text_items]), 
                       "status": status,
                       "meta_items": [checkbox, *corrected_text_items]}
                final_checkbox_group.append(tmp)
                flag_match = True
                skip_idx.append(checkbox["id"])
                for text_item in corrected_text_items:
                    skip_idx.append(text_item["id"])
        if not flag_match:
            single_checkbox.append(checkbox)

    final_single_checkbox = []
    megred_idx = []
    for idxA, checkboxA in enumerate(single_checkbox):
        flag_match = False
        A_bbox = checkboxA["bbox"]
        A_text = checkboxA.get("text", "")
        A_status = checkboxA["status"]
        for idxB, checkboxB in enumerate(single_checkbox):
            if idxA <= idxB:
                continue
            if idxB in megred_idx:
                continue
            B_bbox = checkboxB["bbox"]
            B_text = checkboxB.get("text", "")
            B_status = checkboxB["status"]
            if is_horizontal_overlap(B_bbox, A_bbox):
                if (len(A_text) > 3 or len(B_text) > 3) and not (len(A_text) > 3 and len(B_text) > 3):
                    flag_match = True
                    if B_status[0] == "checked" or A_status[0] == "checked" or A_text.strip().lower() == "on" or B_text.strip().lower() == "on":
                        final_status = "checked"
                    else:
                        final_status = "unchecked"
                    merged_bbox = merge_bbox([checkboxA, checkboxB])
                    tmp = {"category": "check_group", 
                           "bbox":merged_bbox, 
                           "text": ",".join([A_text, B_text]), 
                           "status": [final_status, 1],
                           "meta_items": [checkboxA, checkboxB]}
                    final_single_checkbox.append(tmp)
                    megred_idx.extend([idxA, idxB])
                    skip_idx.append(checkboxA["id"])
                    skip_idx.append(checkboxB["id"])
                    break

    for idx, checkbox in enumerate(single_checkbox):
        if idx in megred_idx:
            continue
        if len(checkbox.get("text", "")) > 3:
            checkbox["category"] = "check_group"
            checkbox["meta_items"] = []
            final_single_checkbox.append(checkbox)
            skip_idx.append(checkbox["id"])
    final_checkbox_group.extend(final_single_checkbox)


    skip_idx.sort(reverse=True)
    for skip in skip_idx:
        del item_list[skip]
    item_list.extend(final_checkbox_group)

    for idx in list(range(len(item_list)))[::-1]:
        # print(idx)
        if item_list[idx]["bbox"][1] <= 20 and item_list[idx].get("iconLabel", [""])[0] not in ["ICON-SMALLCLOSE", "ICON-ADINFO"]:
            del item_list[idx]


    return item_list 