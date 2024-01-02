
import sys, json
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np

from paddleocr import PaddleOCR
paddle_model = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv2")

def rotateImage(image, angle):
    row,col, _ = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def detect_text_paddle(img_cv, output_json, vis=False):
    print('*** Detect Text through Paddle OCR ***')

    img = img_cv.copy()
    # img = img[50:80, 460:528] 
    # img = rotateImage(img, -45)
    result = paddle_model.ocr(img, cls=True)
    reformat_text = []
    for re in result: ## result[0]
        x1 = min([int(a[0]) for a in re[0]])
        x2 = max([int(a[0]) for a in re[0]])
        y1 = min([int(a[1]) for a in re[0]])
        y2 = max([int(a[1]) for a in re[0]])

        tmp_text = re[1][0]
        tep_score = round(re[1][1].item(), 3)
        # tep_score = round(re[1][1], 3)
        # print(tep_score)

        reformat_text.append({"category": "pText",
                              "bbox": [x1,y1,x2,y2],
                              "text": tmp_text,
                              "score": tep_score
                            })
    # img = cv2.imread(input_file)
    if vis:
        draw_ocr(img, reformat_text, output_json)

    with open(output_json, "w") as f:
        json.dump(reformat_text, f)
    return reformat_text


def draw_ocr(img, results, output_json):
    for item in results:
        bbox = item["bbox"]
        text = item["text"]

        img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), 
                                         color=(0,0,255), thickness=5)
        img = cv2.putText(img, text, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=3)
    cv2.imshow("2-paddle ocr", img)
    cv2.waitKey(0)

    result_path = output_json.replace(".json", ".jpg")
    # print(result_path)
    cv2.imwrite(result_path, img)


if __name__ == '__main__':
        
    # all_rico_imgs = glob("/Users/che444/Desktop/DPCode/final_dataset/CHI2020_DP_GT/**/**.jpg")
    # all_rico_imgs = glob("../processed_data/Rico_testset/testset/**.jpg")
    all_rico_imgs = glob("/Users/che444/Desktop/Meeting/Dark Patterns/paper/user_study_data/Step4/**.jpg")
    # all_rico_imgs = glob("**.jpg")
    all_rico_imgs.sort()
    # # get paddle results
    all_results = {}
    for idx, rico_path in enumerate(tqdm(all_rico_imgs)):
        # print(rico_path)
        # if "46067.jpg" not in rico_path:
        #     continue
        text_items = detect_text_paddle(rico_path)
        # print(text_items)
        all_results[rico_path] = text_items

        # if (idx+1) % 300 == 0:
        #     print("==> PADDLE IDX", idx)
        #     with open("all_mobbin_ios_ocr_results_tmp.json", "w") as f:
        #         json.dump(all_results, f)

    # with open("/Users/che444/Desktop/DPCode/final_dataset/CHI2020_DP_GT_ocr_results.json", "w") as f:
    #     json.dump(all_results, f)


