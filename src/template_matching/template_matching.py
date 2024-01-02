import cv2, os,json
import numpy as np
from glob import glob
import re
from tqdm import tqdm
from matplotlib import pyplot as plt


class TemplateMatching:
    def __init__(self):
        self.temp_root_adinfo = "/Users/che444/Desktop/DPCODE-CLEAN/finalCode/template_matching/templates/ADTriInfo"
        self.temp_img_paths_adinfo = glob(self.temp_root_adinfo + "/**.jpg")
        self.all_temp_imgs_adinfo = [cv2.imread(p) for p in self.temp_img_paths_adinfo]
        

        ## AD CLose icon
        self.temp_root_adclose = "/Users/che444/Desktop/DPCODE-CLEAN/finalCode/template_matching/templates/AdClose"
        self.temp_img_paths_adclose = glob(self.temp_root_adclose + "/**.jpg")
        self.all_temp_imgs_adclose = [cv2.imread(p) for p in self.temp_img_paths_adclose]


        #### FOR SIFT
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        self.all_temp_imgs_adinfo_sift = []
        for img in self.all_temp_imgs_adinfo:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("result", grey)
            # cv2.waitKey()
            kp1, des1 = self.sift.detectAndCompute(grey,None)
            # print(des1)
            self.all_temp_imgs_adinfo_sift.append([kp1, des1])

        # create BFMatcher object
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)


    def is_overlapped(self, bbox1, bbox2):
        '''
        check whether two bboxes overlap
        used for ad icon detection
        '''
        x1,y1,x2,y2 = bbox1
        w, y = x2-x1, y2-y1
        xx1,yy1,xx2,yy2 = bbox2
        ww, yy = xx2-xx1, yy2-yy1

        if w+ww > max(abs(x1-xx2), abs(x2-xx1)) and y+yy > max(abs(y1-yy2), abs(y2-yy1)):
            return True
        return False

    def match_template(self, templates, input_img, THRES_SIM = 0.98, THRES_TIME=2):
        '''
        used for detecting ad icons (i.e., info icon, close icon)
        input_img: open by cv2
        '''

        ### multi-scale version can be seen in https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

        exiting_bbox = {}
        for i, templ in enumerate(templates):
            # load template 
            h, w = templ.shape[:-1]

            result = cv2.matchTemplate(input_img, templ, cv2.TM_CCORR_NORMED)
            # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )
          
            _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
            loc = [maxLoc]
            if _maxVal <= THRES_SIM:
                continue
            # print(_maxVal, maxLoc)
            # matchLoc = maxLoc

            # loc = np.where( result > 0.9899)
            # for pt in zip(*loc[::-1]):
            for pt in loc:
                # print(pt)
                flag_break = False
                # print(result[pt[1], pt[0]])
                curr_bbox = [*pt, pt[0] + templ.shape[0], pt[1] + templ.shape[1]]
                for bbox in exiting_bbox.keys():
                    # print(bbox)
                    bbox =  [int(n) for n in re.findall(r"\d+", bbox)]
                    if self.is_overlapped(bbox, curr_bbox):
                        exiting_bbox[str(bbox)][0] += 1
                        exiting_bbox[str(bbox)][1] = max(_maxVal, exiting_bbox[str(bbox)][1])
                        flag_break = True
                        # print("is_overlapped")
                if not flag_break:
                    exiting_bbox[str(curr_bbox)] = [1, _maxVal]
        # print(exiting_bbox)
        output_box = [bbox for bbox, times in exiting_bbox.items() if times[0] >=THRES_TIME]
        output_box22 = []
        for bbox in output_box:
            bbox =  [int(n) for n in re.findall(r"\d+", bbox)]
            output_box22.append(bbox)
        # print("output_box22", output_box22)
        return exiting_bbox, output_box22


    def find_adInfo(self, input_img):
        '''
        used for detecting ad icons (i.e., info icon, close icon)
        input_img: open by cv2
        '''

        ### multi-scale version can be seen in https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

        exiting_bbox, output_box22 = self.match_template(self.all_temp_imgs_adinfo, input_img, THRES_SIM = 0.98, THRES_TIME=2)
        return exiting_bbox, output_box22

    def find_adClose(self, input_img):
        '''
        used for detecting ad icons (i.e., info icon, close icon)
        input_img: open by cv2
        '''

        ### multi-scale version can be seen in https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

        exiting_bbox, output_box22 = self.match_template(self.all_temp_imgs_adclose, input_img, THRES_SIM = 0.995, THRES_TIME=1)
        return exiting_bbox, output_box22

    def find_adInfo_sift(self, input_img):

        img2 = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.sift.detectAndCompute(img2,None)

        all_src_pts = []
        all_dst_pts = []
        for i, (kp1, des1) in enumerate(self.all_temp_imgs_adinfo_sift):
            img1 = self.all_temp_imgs_adinfo[i]

            matches = self.flann.knnMatch(des1,des2,k=2)
            good_matches = []
            for m,n in matches:
                if m.distance < 0.7 *n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                continue
            print(len(good_matches))

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches     ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            print(dst_pts)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            # try:
            dst = cv2.perspectiveTransform(pts,M)
            print(dst)
            x1 = max(0, int(min([a[0][0] for a in dst])))
            y1 = max(0, int(min([a[0][1] for a in dst])))
            x2 = int(max([a[0][0] for a in dst]))
            y2 = int(max([a[0][1] for a in dst]))

            print(x1,y1,x2,y2)

            # Draw bounding box in Red
            img3 = cv2.rectangle(img2, (x1,y1), (x2,y2), color=(0,0,255), thickness=3)
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #                singlePointColor = None,
            #                matchesMask = matchesMask, # draw only inliers
            #                flags = 2)

            # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

            cv2.imshow("result", img3)
            cv2.waitKey()



def get_gt_info():
    json_path = "/Users/che444/Desktop/DPCode/processed_data/Rico_testset/rico_test_annotations.json"
    annos = json.load(open(json_path, "r"))

    ## ng 
    imgid2bboxes = {}
    for img_id, img_dict in annos.items():
        imgid2bboxes[img_id] = []
        # if img_id == "18720.jpg":
        #     print(img_dict)
        if len(img_dict) == 0:
            continue
        instances = img_dict["instances"]
        ng_inst = instances.get("NG", [])
        if len(ng_inst) == 0:
            continue
        for ng in ng_inst:
            children = ng.get("children", [])
            for child in children:
                child_label = child["label"]
                child_bbox = child["bbox"]

                if child_label.upper() == "ICON-ADINFO":
                    imgid2bboxes[img_id].append(child_bbox)
    return imgid2bboxes

def get_gt_close():
    json_path = "/Users/che444/Desktop/DPCode/processed_data/Rico_testset/rico_test_annotations.json"
    annos = json.load(open(json_path, "r"))

    ## ng 
    imgid2bboxes = {}
    for img_id, img_dict in annos.items():
        imgid2bboxes[img_id] = []
        # if img_id == "18720.jpg":
        #     print(img_dict)
        if len(img_dict) == 0:
            continue
        instances = img_dict["instances"]
        ng_inst = instances.get('ICON-SMALLCLOSE', [])
        if len(ng_inst) == 0:
            continue
        for ng in ng_inst:
            ng_bbox = ng["bbox"]
            imgid2bboxes[img_id].append(ng_bbox)
    return imgid2bboxes

def check_overlap(gt_bboxes, pred_bboxes, tm):
    TP, FP, FN = 0, 0, 0
    left_gt_box = []
    pred_box_flag = [0] * len(pred_bboxes)
    # print("gt", gt_bboxes)
    for gt_box in gt_bboxes:
        flag_match = False
        for pred_id, pred in enumerate(pred_bboxes):
            if tm.is_overlapped(gt_box, pred):
                pred_box_flag[pred_id] = 1
                flag_match = True
                TP += 1
                break
        if flag_match is False:
            left_gt_box.append(gt_box)
            FN += 1

    FP += len(pred_bboxes) - sum(pred_box_flag)
    if len(left_gt_box) > 0 or sum(pred_box_flag) != len(pred_bboxes):
        return False, left_gt_box, TP, FP, FN
    else:
        return True, [], TP, FP, FN


def draw_bbox(img, bboxes,color = (0,255,0)):
    for box in bboxes:
        img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color=color, thickness=2)
    return img


def save(img_id, img):
    target_fold = "ad_Close_mobbin_android"
    if not os.path.exists(target_fold):
        os.makedirs(target_fold)
    img_path = os.path.join(target_fold, img_id)
    cv2.imwrite(img_path, img)


tm = TemplateMatching()

def get_ad_icons(img_cv, output_path, vis=False):
    img = img_cv.copy()

    exiting_bbox, pred_bboxes = tm.find_adClose(img)
    exiting_bbox_info, pred_bboxes_info = tm.find_adInfo(img)

    # print(pred_bboxes, pred_bboxes_info)

    # flag, left_gt_bboxes, TPtmp, FPtmp, FNtmp = check_overlap(gt_bboxes, pred_bboxes, tm)

    if vis:
        img = draw_bbox(img, pred_bboxes)
        img = draw_bbox(img, pred_bboxes_info)
        # cv2.imshow("template matching", img)
        # cv2.waitKey()
        cv2.imwrite(output_path.replace(".json", ".jpg"), img)


    with open(output_path, "w") as f:
        json.dump([pred_bboxes, pred_bboxes_info], f)
    return pred_bboxes, pred_bboxes_info
