import os, json
import re
from tqdm import tqdm
from glob import glob
import time

from utils import merge_bbox, is_horizontal_overlap
from DBSCAN.grouping import get_grouping
import cv2
from matplotlib import pyplot as plt
import numpy as np
cmap = plt.get_cmap("tab10")

# ['TextView', 'ImageView', 'RadioButton', 'pText', 'ProgressBar', 'CheckBox', 'ToggleButton', 'EditText', 'ImageButton', 'SeekBar', 'Switch', 'RatingBar', 'Button']

category2eleType = {"ImageView": "Icon", "ImageButton": "Icon", 
                    "TextView": "Text", 'pText': "Text",
                    "Switch": "CheckBox", "CheckBox":"CheckBox",
                    "ToggleButton": "CheckBox",
                    'RadioButton':'RadioButton',
                    "ProgressBar": "ProgressBar",
                    "EditText": "Input",
                    "SeekBar": "SeekBar",
                    "RatingBar": "Rating",
                    "Button": "Button",
                    "check_group": "check_group"
                    }

flag_icon = True
flag_TM = True
flag_status = True
flag_grouping = True

#   [ {"category": pText, 
                    # valid choice: ['TextView', 'ImageView', 'RadioButton', 'pText', 'ProgressBar', 'CheckBox', 'ToggleButton', 'EditText', 'ImageButton', 'SeekBar', 'Switch', 'RatingBar', 'Button']
#      "bbox": [x1,y1,x2,y2],
#      "text": xxxx,
#      "score": float},
#      {"category": TextView,
#       "bbox": [x1,y1,x2,y2],
#       "score": float,
#       "matched": boolean,
#       "text_items": [ {"category": pText,
#                        "bbox": [x1,y1,x2,y2],
#                        "text": xxxx,
#                        "score": float},
#                         {..}]

def get_iou(box_A,box_B):
    # print(box_A, box_B)
    area_d = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    area_gt = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1])
    col_min = max(box_A[0], box_B[0])
    row_min = max(box_A[1], box_B[1])
    col_max = min(box_A[2], box_B[2])
    row_max = min(box_A[3], box_B[3])
    # if not intersected, area intersection should be 0
    w = max(0, col_max - col_min)
    h = max(0, row_max - row_min)
    area_inter = w * h
    if area_inter == 0:
        return False
    iod = area_inter / area_d
    iou = area_inter / (area_d + area_gt - area_inter)

    if iod>0.8 or iou>0.7:
        return True
    return False

def append_item(flags, name, idx):
    if name not in flags:
        flags[name] = []
    flags[name].append(idx)

def examine_text_content(tmp_text, idx, flags, pro_yes=False):
    tmp_text = tmp_text.lower()
    # print(tmp_text)
    ## ad
    # pro
    # pro_yes = False

    pro_pattern = re.compile(r'^(.*)\b(remove|without|disable|block).+(ad|ads|advertisement|advertisements|advertising|adverts)\b') 
    mo1 = pro_pattern.search(tmp_text)

    pro_pattern2 = re.compile(r'^(.*)\b(no ad|no ads|no advertisement|no advertisements|no advertising|noadverts|ad free|no-ad)\b') 
    mo2 = pro_pattern2.search(tmp_text)
    # print(tmp_text)
    if not (mo1 is None) or not (mo2 is None):
        append_item(flags, "flag_pro", idx)
        # print("[PRO]", tmp_text)
        pro_yes = True

    ## watch ad to unlock feature
    watch_pattern = re.compile(r'^(.*)\b(watch).*\b(ad|ads|advertisement|advertisements|advertising|adverts)\b') 
    mo1 = watch_pattern.search(tmp_text)
    if not (mo1 is None) :
        append_item(flags, "flag_watch", idx)
        pro_yes = True


    ad_keywords = ["ads", "ad", "sponsored", "advertisement", "advertisements", "adchoices", "promoted"] #, "google play"]
    tokens =  re.split('[^a-zA-Z]', tmp_text)

    # print(tokens)
    if not pro_yes and len(tmp_text.split())< 6:
        flag_12 = False
        for kw in ad_keywords:
            if kw in tokens:
                append_item(flags, "flag_ad_text", idx)
                # print("[AD TEXT]", tmp_text)
                flag_12 = True
                break

    google_pattern = re.compile(r'^(.*)(get|download|find|app).*(on)(.+)(app store|google play| )(.*)') 
    mo1 = google_pattern.search(tmp_text)
    if not (mo1 is None):
        append_item(flags, "flag_google", idx)


    ## pop up rate
    # print(tmp_text)
    rate_pattern = re.compile(r'^(.*)(do|if).*(you).*(like|enjoy|love)(.+)(rate|star|google play|play store|app store|app)(.*)') 
    mo1 = rate_pattern.search(tmp_text)
    if not (mo1 is None) and len(tmp_text.split()) < 50:
        append_item(flags, "flag_rate", idx)

    ## premium
    free_pattern = re.compile(r'^(.*)(try|get|start|enjoy).*(month|week|day)(.+)(free)(.*)') 
    upgrade_pattern = re.compile(r'(.*)(upgrade|download the prenium|free trial).*(premium|)')
    mo1 = free_pattern.search(tmp_text)
    mo2 = upgrade_pattern.search(tmp_text)
    if not (mo1 is None) or "premium" in tmp_text or "upgrade" in tmp_text.strip() or (not (mo2 is None)) :
        append_item(flags, "flag_premium", idx)
        # print("UPGRAGE", tmp_text)


    # multiple currencies
    if ("coin" in tmp_text.split() or "coins" in tmp_text.split()) and len(tmp_text.split())<5:
        append_item(flags, "flag_coin", idx)
        # print("COIN:", tmp_text)

    real_currency = ["$", "¥", "£", "€", "dollar", "pound"]
    for kw in real_currency:
        if kw in tmp_text:
            append_item(flags, "flag_real_money", idx)
            break


    # if forced continuity: subscription & hard to recognize
    sub_pattern = re.compile(r'^.+(free|free trial).+(then|continue).+($| )\d+.*(year|month|week|forenight).*') 
    mo1 = sub_pattern.search(tmp_text)
    if not (mo1 is None):
        append_item(flags, "flag_sub", idx)
        # print("##", tmp_text)

    ### privacy with checkbox/toggle button
    privacy_pattern = re.compile(r'^(.*)(you|i|).*(consent|agree|give consent|accept)(.+)(terms|terms of use|privacy policy|policies|terms and conditions|terms & conditions|terms of service|license agreement)(.*)') 
    mo1 = privacy_pattern.search(tmp_text)
    if not (mo1 is None) or "acepto las Condiciones de uso" in tmp_text:
        append_item(flags, "flag_pre_privacy", idx)

    if "content notice" in tmp_text or "privacy policy"  in tmp_text:
        append_item(flags, "flag_pre_content_notice", idx)
    if "ok" in tmp_text.split():
        append_item(flags, "flag_pre_ok", idx)

    ### notification with checkbox or toggle button
    notification_pattern = re.compile(r'^(.*)(push|enable|links in|daily|allow|turn on|receive|get).*(notification|newsletter|news|update|message|email).*') 
    mo1 = notification_pattern.search(tmp_text)

    notification_pattern2 = re.compile(r'^(.*)(notify|notification|newsletter|news).*') #update|message|email|ringtone|sound|vibration|vibrate
    mo2 = notification_pattern2.search(tmp_text)

    if (not (mo2 is None)) or (not (mo1 is None)):
        append_item(flags, "flag_pre_notification", idx)


    ## usage data
    usage_data_pattern = re.compile(r'^(.*)(usage tracking|analytics|usage data|usage statistics|usage report|collect information|analytics purpose|anonymous stats|usage info).*') 
    mo1 = usage_data_pattern.search(tmp_text)
    if not (mo1 is None):
        append_item(flags, "flag_pre_usage_data", idx)

    ## follow 
    if "follow" in tmp_text:
        append_item(flags, "flag_pre_follow", idx)

    ## no checkbox
    privacy_pattern = re.compile(r'^(.*)(by| )(.*)(continuing|tapping|using|creating|signing|clicking|pressing|logging|installing|logging|taping).*(allow|agree|consent|accept|confirm)(.*)')  # (terms of service|terms of use|privacy policies|privacy policy|terms|terms and conditions|user agreement|terms of user agreement|cookie policy|use policy|terms & conditions|privacy statement|license agreement|rule| )(.*)
    mo1 = privacy_pattern.search(tmp_text)
    if not (mo1 is None) and len(tmp_text.split()) <100:
        append_item(flags, "flag_pre_privacy_by", idx)
        # print(mo1)
    if "signing" in tmp_text:
        print(tmp_text)

    ## countdown
    free_pattern = re.compile(r'(.*) (reward|offer).* (end|expire).*(at|in)(.*)') 
    mo3 = free_pattern.search(tmp_text)
    if not (mo3 is None):
        append_item(flags, "flag_offer", idx)


    countdown_pattern = re.compile(r'^(.*) (second|s|seconds) .*(remaining)(.*)(.*)') 
    mo1 = countdown_pattern.search(tmp_text)

    countdown_pattern = re.compile(r'^(.*)\d+:\d+(.*)') 
    mo2 = countdown_pattern.search(tmp_text)
    if (not (mo1 is None)) or (not (mo2 is None)):
        append_item(flags, "flag_countdown", idx)


    # ad countdown

    ad_countdown_pattern = re.compile(r'^(.*)\bclick to skip ad.*\d+.*s.*') 
    mo2 = ad_countdown_pattern.search(tmp_text)
    if (not (mo1 is None)) or (not (mo2 is None)):
        append_item(flags, "flag_countdown_ad", idx)


    # socialspam
    friend_pattern = re.compile(r'^(.*)(refer|invite).*(friend)(.*)') 
    mo1 = friend_pattern.search(tmp_text)
    if not (mo1 is None) :
        append_item(flags, "flag_friend", idx)



    ## can skip
    if "skip" in tmp_text:
        append_item(flags, "flag_skip", idx)

    ## ad continue patterns
    ad_continue_patterns = ["no thanks", "no thank", "continue", "exit", "next time", "not for now", "later"]

    for ad_continue in ad_continue_patterns:
        if tmp_text.strip() == "no":
            append_item(flags, "flag_ad_continue", idx)
            break

        if ad_continue in tmp_text:
            append_item(flags, "flag_ad_continue", idx)
            break


    ## daily rewards
    daily_pattern = re.compile(r'^(.*)(daily|weekly|nightly).*(reward|bonus|dozon)(.*)(.*)') 
    mo1 = daily_pattern.search(tmp_text)
    if not (mo1 is None):
        append_item(flags, "flag_daily", idx)


def examine_icons(iconLabel, idx, flags):
    # print(iconLabel)
    if flag_icon:
        if iconLabel == "star":
            append_item(flags, "flag_star_icon", idx)
        elif iconLabel == "close":
            append_item(flags, "flag_close_icon", idx)
        elif iconLabel in ["info"]:
            append_item(flags, "flag_info_icon", idx)
        elif iconLabel == "play":
            append_item(flags, "flag_adPlay_icon", idx)

    if flag_TM:
        if iconLabel == "ICON-SMALLCLOSE":
            append_item(flags, "flag_smallClose_icon", idx)
        elif iconLabel == "ICON-ADINFO":
            append_item(flags, "flag_adInfo_icon", idx)

    # elif iconLabel in ["play", "info"]:
    #     append_item(flags, "flag_ad_icon", idx)


 
def examine_status(status, idx, flags):
    # print(status)
    if status == "checked":
        append_item(flags, "flag_check", idx)
    elif status == "unchecked":
        append_item(flags, "flag_uncheck", idx)


def examine_color(colors, flags):
    pass


def examine_size(colors, flags):
    pass

def examine_compound(item, idx, flags):
    item_category = item["category"]
    item_text = item.get("text", "").lower()
    item_bbox = item["bbox"]
    iconLabel = item.get("iconLabel", [""])[0]


    ## II-AM-DA buttons/icons are ad but not clear
    # print(item_text.split(), "ad" in item_text.split(), item_category)
    if item_category in ["ImageView", "ImageButton"] and item_text.strip() in ["ad", "ads"]:
        # if not get_iou(item["text_items"][0]["bbox"], item["bbox"]):
            append_item(flags, "flag_notClearAd", idx)
            # print("asda")
            return True
    return False

    ## AM - general - small/greyed text
    ## TODO


def get_info(item_list, item_ids, subType, flag):
    tmp_re = []
    for item_id in item_ids:
        item = item_list[item_id]
        item_category = item["category"]
        item_bbox = item["bbox"]

        tmp_info = { "type": category2eleType[item_category],
                        "bbox": item_bbox,
                        "subType": subType,
                        "flag": flag,
                        "text": item.get("text", ""),
            }
        if "status" in item:
            tmp_info["status"] = item["status"]

        tmp_re.append(tmp_info)
    return tmp_re

def check_naggings(flags, item_list, output_results):
    ### POP-UP AD
    # for tmp_flag in ["flag_ad_text", "flag_google", "flag_adInfo_icon"]:
    #     if tmp_flag in flags:
    #         if "NG" not in output_results:
    #             output_results["NG"] = []
    #         tmp_re = get_info(item_list, flags[tmp_flag], "[NG] Pop-up Ad")
    #         output_results["NG"].append(tmp_re)

    ## POP-UP to RATE
    for tmp_flag in ["flag_rate"]:
        if tmp_flag in flags:
            if "NG-RATE" not in output_results:
                output_results["NG-RATE"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[NG] A rating page pops up unexpectedly, interrupt user tasks and nags users.", tmp_flag)
            output_results["NG-RATE"].extend(tmp_re)

    ## POP-UP premium
    for tmp_flag in ["flag_premium"]:
        if tmp_flag in flags:
            if "NG-UPGRADE" not in output_results:
                output_results["NG-UPGRADE"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[NG] An ungrade page unexpectedly pops up and nags users to upgrade to premium version", tmp_flag)
            output_results["NG-UPGRADE"].extend(tmp_re)

def check_obstruction(flags, item_list, output_results):
    ## multiple currency
    for tmp_flag in ["flag_coin", "flag_real_money"]:
        if tmp_flag in flags:
            if "OB-IntCur" not in output_results:
                output_results["OB-IntCur"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[OB-IntCur] Intermediate Currency", tmp_flag)
            output_results["OB-IntCur"].extend(tmp_re)


def check_sneaking(flags, item_list, output_results):
    ## forced continuity
    for tmp_flag in ["flag_sub"]:
        if tmp_flag in flags:
            if "SN-FC" not in output_results:
                output_results["SN-FC"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[SN-FC] Forced Continuity - The app provides a free trial, but it forces users to consent to auto-subscribe after the end of the free trial. ", tmp_flag)
            output_results["SN-FC"].extend(tmp_re)


def check_interface_inference(flags, item_list, output_results):
    ## hidden information
    pass

    ## Preselection

    for tmp_flag in ["flag_pre_privacy_by"]:
        if tmp_flag in flags:
            if "II-PRE-PRIVACY-BY" not in output_results:
                output_results["II-PRE-PRIVACY-BY"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] Users are forced to consent to privacy policies/terms of use as there is no checkbox", tmp_flag)
            output_results["II-PRE-PRIVACY-BY"].extend(tmp_re)

    for tmp_flag in ["flag_pre_privacy"]:
        if tmp_flag in flags:
            if "II-PRE-PRIVACY" not in output_results:
                output_results["II-PRE-PRIVACY"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] The privacy policy/terms of use is consent by default", tmp_flag)
            output_results["II-PRE-PRIVACY"].extend(tmp_re)

    for tmp_flag in ["flag_pre_content_notice"]:
        if tmp_flag in flags:
            if "II-PRE-CONTENT" not in output_results:
                output_results["II-PRE-CONTENT"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] The privacy policy/terms of use is consent by default", tmp_flag)
            output_results["II-PRE-CONTENT"].extend(tmp_re)

    for tmp_flag in ["flag_pre_notification"]:
        if tmp_flag in flags:
            if "II-PRE-NOTIFICATION" not in output_results:
                output_results["II-PRE-NOTIFICATION"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] Enable notification by default", tmp_flag)
            output_results["II-PRE-NOTIFICATION"].extend(tmp_re)

            # print("##item", tmp_re)

    for tmp_flag in ["flag_pre_usage_data"]:
        if tmp_flag in flags:
            if "II-PRE-USAGE-DATA" not in output_results:
                output_results["II-PRE-USAGE-DATA"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] The app automatically sends users' usage data for analysis by defauly", tmp_flag)
            output_results["II-PRE-USAGE-DATA"].extend(tmp_re)

    for tmp_flag in ["flag_pre_follow"]:
        if tmp_flag in flags:
            if "II-PRE-FOLLOW" not in output_results:
                output_results["II-PRE-FOLLOW"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] Follow some accounts by default", tmp_flag)
            output_results["II-PRE-FOLLOW"].extend(tmp_re)

    # for tmp_flag in ["flag_check", "flag_pre_ok", "flag_uncheck"]:
    #     if tmp_flag in flags:
    #         if "II-PRE-CHECKBOX" not in output_results:
    #             output_results["II-PRE-CHECKBOX"] = []
    #         tmp_re = get_info(item_list, flags[tmp_flag], "[PRE] CHECKBOX ", tmp_flag)
    #         output_results["II-PRE-CHECKBOX"].extend(tmp_re)

    ## AM - Toying with emotion
    for tmp_flag in ["flag_offer"]: #"flag_emoji", 
        if tmp_flag in flags:
            if "II-AM-TWE" not in output_results:
                output_results["II-AM-TWE"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[II-AM-TWE] Countdown Offer/Rewards to make users nervous", tmp_flag)
            output_results["II-AM-TWE"].extend(tmp_re)


    ## AD texts/icons for examing DA, NG-AD, FA-CountdownAD
    for tmp_flag in ["flag_ad_text", "flag_google", "flag_adInfo_icon", "flag_close_icon", "flag_countdown", "flag_ad_continue", "flag_countdown_ad", "flag_adPlay_icon", "flag_info_icon"]:
        if tmp_flag in flags:
            if "AD" not in output_results:
                output_results["AD"] = []
            tmp_re = get_info(item_list, flags[tmp_flag], "[AD] Ad Icons/Texts", tmp_flag)
            output_results["AD"].extend(tmp_re)
            

    ## AM - Disguised AD
    ## "[DA] Sponsored Content looks like normal content"


    ## [DA] Icons/Buttons are ads, but noe clear
    tmp_flag = "flag_notClearAd"
    if "flag_notClearAd" in flags:
        if "II-AM-DA" not in output_results:
            output_results["II-AM-DA"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[DA] Icon/Button is ad, but it is not clear", "flag_notClearAd")
        output_results["II-AM-DA"].extend(tmp_re)

    ## need to refine
    ## [DA] Ad with Interactive Game


    ## AM - General Type
    ### [AM] Small Close Button
    tmp_flag = "flag_smallClose_icon"
    if "flag_smallClose_icon" in flags:
        if "II-AM-G-SMALL" not in output_results:
            output_results["II-AM-G-SMALL"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[AM] The close icon in ad is so small that users may mistakenly open the ad.", tmp_flag)
        output_results["II-AM-G-SMALL"].extend(tmp_re)


def check_forced_action(flags, item_list, output_results):

    # social pyramid
    tmp_flag = "flag_friend"
    if "flag_friend" in flags:
        if "FA-SOCIALPYRAMID" not in output_results:
            output_results["FA-SOCIALPYRAMID"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[FA] Users can obtain some rewards by invitating their friends, which puts social pressure to their friends", "flag_friend")
        output_results["FA-SOCIALPYRAMID"].extend(tmp_re)

    ## FA - general type - watch apps to u
    tmp_flag = "flag_watch"
    if "flag_watch" in flags:
        if "FA-G-WATCHAD" not in output_results:
            output_results["FA-G-WATCHAD"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[FA] Users are forced to watch Ads to unlock some features", "flag_watch")
        output_results["FA-G-WATCHAD"].extend(tmp_re)

    ## FA - general type - [PRO] pay to avoid ads
    tmp_flag = "flag_pro"
    if "flag_pro" in flags:
        if "FA-G-PRO" not in output_results:
            output_results["FA-G-PRO"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[FA-G-PRO] Users have to pay to avoid ads", "flag_pro")
        output_results["FA-G-PRO"].extend(tmp_re)

    ## FA GAMIFICATION
    tmp_flag = "flag_daily"
    if "flag_daily" in flags:
        if "FA-GAMIFICATION" not in output_results:
            output_results["FA-GAMIFICATION"] = []
        tmp_re = get_info(item_list, flags[tmp_flag], "[FA-GAMIFICATION] daily awards", "flag_daily")
        output_results["FA-GAMIFICATION"].extend(tmp_re)


def add_to_final_results(final_, typ, data):
    if typ not in final_:
        final_[typ] = []
    final_[typ].append(data)

def final_check(output_results, img_h, img_w, all_items):
    # tmp_re.append({ "type": category2eleType[item_category],
    #                     "bbox": item_bbox,
    #                     "subType": subType,
    #                     "flag": flag,
    #         })

    final_results = {}
    # print(output_results)



    for typ in ["II-AM-G-SMALL", "FA-G-WATCHAD",  "SN-FC", "II-AM-TWE", "II-AM-DA"]:
        typ_items = output_results.get(typ, [])

        if len(typ_items) == 0:
            continue
        final_results[typ] = []
        for item in typ_items:
            # print("##item", typ_items, typ)
            data = {"label":typ, 
                    "bbox":item["bbox"], 
                    "subType": item["subType"], 
                    "flag": item["flag"]}
            add_to_final_results(final_results, typ, data)


    for typ in ["FA-SOCIALPYRAMID", "FA-GAMIFICATION", "NG-RATE", "FA-G-PRO"]:
        typ_items = output_results.get(typ, [])
        if len(typ_items) == 0:
            continue

        children = []
        for itt in typ_items:
            children.append({"label":typ, "bbox":itt["bbox"],  "flag": itt["flag"]})

        merged_bbox = merge_bbox(typ_items)
        data = {"label":typ, 
                "bbox":merged_bbox, 
                "subType":typ_items[0]["subType"], 
                "children": children}
        add_to_final_results(final_results, typ, data)


    typ = "NG-UPGRADE"
    if typ in output_results:
        typ_items = output_results.get(typ, [])
        if len(typ_items) > 0:
            children = []
            for itt in typ_items:
                children.append({"label":typ, "bbox":itt["bbox"],  "flag": itt["flag"]})

            merged_bbox = merge_bbox(typ_items)
            x1,y1,x2,y2 = merged_bbox
            x_center = (x1+x2) //2
            y_center = (y1+y2) //2
            if img_w//3 <= x_center <= 2*img_w//3 and img_h//3 <=y_center <= 2*img_h//3:
            # if True:
                data = {"label":typ, 
                        "bbox":merged_bbox, 
                        "subType":typ_items[0]["subType"], 
                        "children": children}
                add_to_final_results(final_results, typ, data)


    typ = "OB-IntCur"
    if typ in output_results:
        typ_items = output_results.get(typ, [])
        if len(typ_items) > 0:
            all_flags = [item["flag"] for item in typ_items]
            if "flag_coin" in all_flags and "flag_real_money" in all_flags:

                children = []
                for itt in typ_items:
                    children.append({"label":typ, "bbox":itt["bbox"],  "flag": itt["flag"]})

                merged_bbox = merge_bbox(typ_items)
                final_results[typ] = [{"label":typ, "bbox":merged_bbox, "subType":typ_items[0]["subType"], "children": children}]


    ### ---- PRESELECTION  ----- ###
    types = ["II-PRE-FOLLOW", "II-PRE-NOTIFICATION", "II-PRE-PRIVACY", "II-PRE-USAGE-DATA", "II-PRE-FOLLOW"]
    item_sorted = sorted(all_items, key= lambda x:[x["bbox"][1], x["bbox"][0]])
    flag_notification_block = False
    noti_block_bbox = None
    included_bbox = []
    for item in item_sorted:
        item_text = item.get("text", "").lower()
        item_cate = item["category"]
        # print("===>", item_cate, item_text, item.get("iconLabel", ""))
        if len(item_text) == 0:
            continue
        if item_cate != "check_group":
            if "notification" in item_text or "usage" in item_text:
                flag_notification_block = True
                noti_block_bbox = item["bbox"]
                # print("++ check block!", item["text"])
            else:
                if flag_notification_block and (not is_horizontal_overlap(noti_block_bbox, item["bbox"])) :
                    flag_notification_block = False 
        else:
            # print(flag_notification_block, "page check:", item["text"], item["status"])
            if flag_notification_block:
                if item["category"] == "check_group":
                    # print("page check:", item)
                    if flag_status:
                        if item["status"][0] == "checked":
                            data = {"label": typ, 
                                    "bbox":item["bbox"], 
                                    "subType": "[II-PRE] The notification is enabled by default",
                                    "flag": "nofication block"}
                            add_to_final_results(final_results, "II-PRE", data)
                            included_bbox.append(item["bbox"])

    for typ in types:
        if typ in output_results:
            typ_items = output_results.get(typ, [])
            final_results[typ] = []
            for item in typ_items:
                if item["bbox"] in included_bbox:
                    continue
                # print(item)
                if item["type"] == "check_group":
                    # print(item)
                    if flag_status:
                        if item["status"][0] == "checked":
                            data = {"label": typ, 
                                    "bbox":item["bbox"], 
                                    "subType":item["subType"],
                                    "flag": item["flag"]}
                            add_to_final_results(final_results, "II-PRE", data)

                            if typ in ["II-PRE-PRIVACY", "II-PRE-USAGE-DATA"]:
                                data = {"label": typ, 
                                        "bbox":item["bbox"], 
                                        "subType":"[FA-Privacy] Privacy related dark patterns",
                                        "flag": item["flag"]}
                                add_to_final_results(final_results, "FA-Privacy", data)


    typ = "II-PRE-PRIVACY-BY"
    if typ in output_results:
        typ_items = output_results.get(typ, [])

        for item in typ_items:
            if item["type"] != "check_group":
                data = {"label":typ, 
                        "bbox":item["bbox"], 
                        "subType": "[II-PRE] Users are forced to consent to privacy policies/terms of use as there is no checkbox",
                        "flag": item["flag"]}
                add_to_final_results(final_results, "II-PRE-Nocheckbox", data)

                data = {"label":typ, 
                        "bbox":item["bbox"], 
                        "subType": "[FA-Privacy] Users are forced to consent to privacy policies/terms of use as there is no checkbox",
                        "flag": item["flag"]}
                add_to_final_results(final_results, "FA-Privacy", data)
            elif item["type"] == "check_group":
                if flag_status:
                    if item["status"][0] == "checked":
                        data = {"label": typ, 
                                "bbox":item["bbox"], 
                                "subType": "[II-PRE] Privacy policy is consented by default",
                                "flag": item["flag"]}
                        add_to_final_results(final_results, "II-PRE", data)

                        data = {"label": typ, 
                                "bbox":item["bbox"], 
                                "subType": "[FA-Privacy] Privacy policy is consented by default",
                                "flag": item["flag"]}
                        add_to_final_results(final_results, "FA-Privacy", data)



    ###  AD ####
    typ = 'AD'
    if "AD" in output_results:
        typ_items = output_results.get(typ, [])

        AD_TEXT_ITEMS = [item for item in typ_items if item["flag"] == "flag_ad_text"]
        # GOOGLE_ITEMS = [item for item in typ_items if item["flag"] == "flag_google"]
        AD_INFO_ITEMS = [item for item in typ_items if item["flag"] in ["flag_adInfo_icon", "flag_info_icon"]]
        AD_Play_ITEMS = [item for item in typ_items if item["flag"] == "flag_adPlay_icon"]
        CLOSE_ITEMS = [item for item in typ_items if item["flag"] in ["flag_close_icon", "flag_ad_continue"]]
        COUNTDOWN_ITEMS = [item for item in typ_items if item["flag"] in ["flag_countdown", "flag_countdown_ad"]]

        # print(typ_items, AD_TEXT_ITEMS)
        flag_ng_ad = False
        potential_ng_ads = []
        for item in [*AD_TEXT_ITEMS, *AD_INFO_ITEMS, *AD_Play_ITEMS]:
            flag_top_right = True
            flag_bottom_left = True
            ## check location
            ad_bbox = item["bbox"]
            x1,y1,x2,y2 = ad_bbox

            if item["flag"]=="flag_adPlay_icon":
                ad_w = x2-x1
                ad_h = y2-y1

                if ad_w > 50 or ad_h>50: continue

            if x1 >= img_w//2:
                flag_bottom_left = False
            else:
                flag_top_right = False
            if item["flag"] == "flag_info_icon":
                flag_top_right = False

            # print("flags", flag_top_right, flag_bottom_left)
            # NAGGING AD
            for screen_item in all_items:
                if (not flag_bottom_left) and (not flag_top_right):
                    break
                screen_bbox = screen_item["bbox"]
                if screen_bbox == ad_bbox:
                    continue
                if screen_item.get("iconLabel", [None, None])[0] == "close":
                    continue

                if flag_top_right:
                    if y1 > screen_bbox[1]-10 and screen_bbox[2]>=img_w//3:
                        flag_top_right = False
                        print("flags", flag_top_right, flag_bottom_left)
                        
                if flag_bottom_left:
                    if y2 < screen_bbox[3]+10 and screen_bbox[0]<=img_w//3:
                        flag_bottom_left = False
                        print("flags", flag_top_right, flag_bottom_left)

                
            # print("flags", flag_top_right, flag_bottom_left)

            if flag_top_right or flag_bottom_left:
                if not ("FA-G-PRO" in final_results and item["flag"] == "flag_ad_text") \
                        and len(output_results.get("II-AM-G-SMALL", [])) == 0:
                    # print("hello")
                    data = {"label": typ, 
                            "bbox": [0,0,img_w, img_h], 
                            "subType": "[NG-AD] Pop up ads. An Ad unexpectedly pops up and interrupts user tasks.",
                            "flag": item["flag"],
                            "children": [item]}
                    # print("NG-AD", item)
                    add_to_final_results(final_results, "NG-AD", data)
                    flag_ng_ad = True

                # COUNTDOWN AD  -- NO close button or Have CountDOwn
                ### close icons are not detected !! 
                if len(COUNTDOWN_ITEMS)>0: #or len(CLOSE_ITEMS) == 0 
                    data = {"label": typ, 
                            "bbox": [0,0,img_w, img_h], 
                            "subType": "[FA-G-COUNTDOWNAD] The user is forced to watch the countdown ad before they could close it.",
                            "flag": item["flag"],
                            "children": [item]}
                    if len(COUNTDOWN_ITEMS) > 0:
                        data["children"].extend(COUNTDOWN_ITEMS)
                    add_to_final_results(final_results, "FA-G-COUNTDOWNAD", data)
                    
                    flag_ng_ad = True


            ## Disguised AD: look like  normal content
            # print("==", item)
            if item["flag"] not in ["flag_adPlay_icon", "flag_info_icon"] \
                and not flag_ng_ad \
                and len(output_results.get("II-AM-G-SMALL", [])) == 0:
                if item["flag"] == "flag_notClearAd":
                    continue
                elif y1 > img_h * 240/1920  and  y2 < img_h * (1920-398)/1920:
                    potential_ng_ads.append(item)

        # merge same line ad
        if len(potential_ng_ads) > 0:
            potential_ng_ads = sorted(potential_ng_ads, key=lambda x:x["bbox"][1])
            merged_same_line_ad = [[]] * len(potential_ng_ads)
            merged_same_line_ad[0].append(potential_ng_ads[0])
            for idx in range(len(potential_ng_ads[1:])):
                if is_horizontal_overlap(potential_ng_ads[idx-1]["bbox"], potential_ng_ads[idx]["bbox"]):
                    aaa = merged_same_line_ad[idx-1]
                    if len(aaa) == 1:
                        if isinstance(aaa, int):
                            merged_same_line_ad[aaa].append(potential_ng_ads[idx])
                            merged_same_line_ad[idx] = aaa
                        else:
                            merged_same_line_ad[idx-1].append(potential_ng_ads[idx])
                            merged_same_line_ad[idx] = idx-1
                    else:
                        merged_same_line_ad[idx-1].append(potential_ng_ads[idx])
                        merged_same_line_ad[idx] = idx-1
                else:
                    merged_same_line_ad[idx].append(potential_ng_ads[idx])

            for items in merged_same_line_ad:
                if isinstance(items, int):
                    continue
                data = {"label":typ, 
                        "bbox":items[0]["bbox"], 
                        "subType": "[II-AM-DA] An advertisement pretends to be a normal content and users may click without knowing it is an ad.",
                        "flag": items[0]["flag"],
                        "children": items[1:]}
                add_to_final_results(final_results, "II-AM-DA", data)

        ## COUNTDOWN OFFER
        # PASS
    return final_results



def check_AM_FH(final_results, item_list, img_h, img_w):
    am_fh_dets = get_grouping(item_list, img_h, img_w)
    for gid, idxes in am_fh_dets.items():

        tmp_items = [item_list[idx] for idx in idxes]
        merged_bbox = merge_bbox(tmp_items)

        data = {"label": "II-AM-FH", 
                "bbox": merged_bbox, 
                "subType": "[II-AM-FH] The option that favors the interest of app provider is more prominent.",
                "flag": "flag_am_fh",
                "children": tmp_items}
        add_to_final_results(final_results, "II-AM-FH", data)
    return final_results


def predict_type(all_properties, img_h, img_w, vis_save_root, img_path, vis=False):
    # meta: gather info of an image
    item_list = all_properties

    flags = {}
    for idx, item in enumerate(item_list):
        item_text = item.get("text", "")
        item_category = item["category"]
        # print("item category:", item_category)
        # icons..
        flag_yes = False
        flag_yes = examine_compound(item, idx, flags)
        # print(flags)

        if len(item_text) > 0:
            examine_text_content(item_text, idx, flags, flag_yes)

        if item_category in ["ImageView", "ImageButton"]:
            examine_icons(item.get("iconLabel", [""])[0], idx, flags)

        # if item_category in ["CheckBox", "Switch", "ToggleButton"]:
        #     examine_status(item.get("status",["unchecked"])[0], idx, flags)

    output_results = {}
    check_naggings(flags, item_list, output_results)
    check_sneaking(flags, item_list, output_results)
    check_obstruction(flags, item_list, output_results)
    check_interface_inference(flags, item_list, output_results)
    check_forced_action(flags, item_list, output_results)

    final_results = final_check(output_results, img_h, img_w, item_list)

    ## add AM results
    if flag_grouping:
        # print("checking grouping")
        final_results = check_AM_FH(final_results, item_list, img_h, img_w)


    if vis:
        draw_results(img_path, final_results, item_list, vis_save_root)
    return final_results


def draw_circle(number):
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img.fill(255)

    CENTER = (64, 64)

    cv2.circle(img, CENTER, 48, (0,0,255), -1)

    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1.5
    TEXT_THICKNESS = 7
    TEXT = str(number)

    text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)

    cv2.putText(img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)
    return img



cv2.namedWindow("main", cv2.WINDOW_NORMAL)
def draw_results(img_path, output_results, item_list, vis_save_root):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    combined_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
    combined_img.fill(255)

    combined_img[:h, :w] = img
    combined_img = cv2.putText(combined_img, os.path.basename(img_path), (w+5, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=3)

    # for item in item_list:
    #     bbox = item["bbox"]
    #     print(bbox, item.get("text", ""), item.get("category", ""), item.get("iconLabel", ""))
    #     combined_img = cv2.rectangle(combined_img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), 
    #                                      color=(284,284,284), thickness=5)

    thickness = 4
    TextScale = 2
    line_gap = 55
    circle_d = 80
    diff = 30
    num_text_per_line = 28
    if h < 1000:
        thickness = 4
        TextScale = 1
        line_gap = 40
        num_text_per_line = 20
        # diff = 30

    count = 1
    color_count = 0
    # print(len(output_results))
    bbox2color = {}
    ins_id = 0
    x_offset = 0
    for DP_type, instances in output_results.items():
        if len(instances) == 0:
            continue

        for idx,item in enumerate(instances):
            curr_color = cmap(color_count % 10)
            curr_color = [int(255*a) for a in curr_color][:3]
            # print("inst", inst)
            # for item in inst:
            x1,y1,x2,y2 = item["bbox"]
            subtype = item["subType"]
            # print(subtype, item["bbox"])
            if str(item["bbox"]) in bbox2color:
                curr_color = bbox2color[str(item["bbox"])]
                x_offset = circle_d
            else:
                bbox2color[str(item["bbox"])] = curr_color
            ins_id += 1


            if "children" in item:
                for child in item["children"]:
                    child_bbox = child["bbox"]
                    combined_img = cv2.rectangle(combined_img, (child_bbox[0],child_bbox[1]), (child_bbox[2],child_bbox[3]), 
                                         color=curr_color, thickness=5)
                    # print("child", child)

            combined_img = cv2.rectangle(combined_img, (x1,y1), (x2,y2), 
                                         color=curr_color, thickness=5)
            count += 1

            ## get circle
            circle = draw_circle(str(ins_id))
            circle = cv2.resize(circle, (circle_d,circle_d), interpolation = cv2.INTER_AREA)
            combined_img[max(0, y1-circle_d): max(0, y1-circle_d)+circle_d, max(0, x1-circle_d)+x_offset: max(0, x1-circle_d)+circle_d+x_offset] =  circle
            x_offset = 0

            combined_img[count*line_gap-diff: count*line_gap+(circle_d-diff), w+3:w+3+circle_d] = circle

            # cv2.imshow("main", circle)
            # cv2.waitKey(0)
            subtype = subtype.split("]")[1]
            if y2-y1 > 0.7*h and x2-x1 > 0.7*w:
                subtype = "[Whole UI]" + subtype


            text_lines = len(subtype) // num_text_per_line + 1
            for text_line_id in range(text_lines):
                curr_text = subtype[text_line_id*num_text_per_line:(text_line_id+1)*num_text_per_line]
                # print(curr_text[-1], text_line_id, subtype[(text_line_id+1)*num_text_per_line])
                if curr_text[-1] != " " and text_line_id != text_lines-1 and subtype[(text_line_id+1)*num_text_per_line] != " ":
                    curr_text+="-"
                combined_img = cv2.putText(combined_img, curr_text, (w+30+40, 30+count*line_gap), cv2.FONT_HERSHEY_SIMPLEX, fontScale=TextScale, color=curr_color, thickness=thickness)
                count += 1
            color_count += 1
    if ins_id == 0:
        combined_img = cv2.putText(combined_img, "No detected malicious UI", (w+30+40, 30+count*line_gap), cv2.FONT_HERSHEY_SIMPLEX, fontScale=TextScale, color=(0,0,255), thickness=thickness)

    vis_path = os.path.join(vis_save_root, os.path.basename(img_path).split(".")[0]+"-dp.jpg")
    # print(vis_path)
    cv2.imwrite(vis_path, combined_img)
    # if count != 1:
    # cv2.imshow("main", combined_img)
    # cv2.waitKey(0)




                   

