import os
import cv2
import sys
import json
from PIL import Image
from tqdm import tqdm
import cv2
import random, time

# sys.path.insert(0, "iconModel")
# sys.path.insert(0, "statusModel")
from iconModel.get_iconLabel import predict_label
from statusModel.get_status import predict_status
from ColorExtraction.get_color import extract_color

print("Load")

# fg_color, fg_lum, bg_color, bg_lum, con = extract_color(PIL_img)

## Format
# imageid: (e.g., "123.jpg")
# 	[ {"category": pText, 
					# valid choice: ['TextView', 'ImageView', 'RadioButton', 'pText', 'ProgressBar', 'CheckBox', 'ToggleButton', 'EditText', 'ImageButton', 'SeekBar', 'Switch', 'RatingBar', 'Button']

# 	   "bbox": [x1,y1,x2,y2],
# 	   "text": xxxx,
# 	   "score": float},

# 	   {"category": TextView,
# 	    "bbox": [x1,y1,x2,y2],
# 	    "score": float,
# 	    "matched": boolean,
# 	    "text_items": [ {"category": pText,
# 					     "bbox": [x1,y1,x2,y2],
# 					     "text": xxxx,
# 					     "score": float},
# 					      {..}]
# 	    }
# 	]

def get_color_status_icon(dets, img_path, output_path):
	# Input: merged detection results
	## Step 1 read image
	PIL_img = Image.open(img_path).convert('RGB')

	icon_needtoLabel = []
	checkbox_needtoLabel = []
	start_11 = time.time()
	for idx, item in enumerate(dets.copy()):
		item_bbox = item["bbox"]
		item_type = item["category"]
		crop_item_img = PIL_img.crop(tuple(item_bbox))

		## Step 3.1 get color
		item_text_items = item.get("text_items", [])
		color_bounds = None
		if len(item_text_items) > 0:
			max_text_len = 0
			for text_item in item_text_items:
				text_bbox = text_item["bbox"]
				text_content = text_item["text"]
				if len(text_content) > max_text_len: 
					color_bounds = text_bbox
					max_text_len = len(text_content)
		else:
			color_bounds = item_bbox

		fg_color, fg_lum, bg_color, bg_lum, con = extract_color(crop_item_img)
		dets[idx]["fg_color"] = fg_color
		dets[idx]["fg_lum"] = fg_lum
		dets[idx]["bg_color"] = bg_color
		dets[idx]["bg_lum"] = bg_lum
		dets[idx]["con"] = con

		## Step3.2 get checkbox status
		if item_type in ['CheckBox', 'ToggleButton', 'Switch']:
			# crop_item_img.show()
			# status = predict_status([crop_item_img])[0]
			# dets[idx]["status"] = status
			# print(status)

			checkbox_needtoLabel.append([idx, crop_item_img])

		## Step 3.3. get icon semantic
		if item_type in ["ImageButton", "ImageView"]:
			# crop_item_img.show()

			icon_needtoLabel.append([idx, crop_item_img])			
			# label = predict_label([crop_item_img])[0]
			# dets[idx]["iconLabel"] = label
			# print(label)
	# print("Color using", time.time() - start_11)
	# start_11 = time.time()
	# batch status
	if len(checkbox_needtoLabel) > 0:
		status_imgs = [item[1] for item in checkbox_needtoLabel]
		status = predict_status(status_imgs)
		for idx, ss in enumerate(status):
			# checkbox_needtoLabel[idx][1].show()
			dets[checkbox_needtoLabel[idx][0]]["status"] = ss
			# print("-->", ss)

	# print("Checkbox using", time.time() - start_11)
	# start_11 = time.time()

	# batch icons
	if len(icon_needtoLabel) > 0:
		icon_imgs = [item[1] for item in icon_needtoLabel]
		labels = predict_label(icon_imgs)
		for idx, ll in enumerate(labels):
			# icon_needtoLabel[idx][1].show()
			dets[icon_needtoLabel[idx][0]]["iconLabel"] = ll
			# print(ll)
	# print("Icon using", time.time() - start_11)
	# start_11 = time.time()

	json.dump(dets, open(output_path, "w"))
	return dets

