# faster rcnn
import sys, json
sys.path.append("elementDetection/FASTER_RCNN/lib")
from model.roi_layers import nms

import torch
from torch.autograd import Variable
import numpy as np

def xywh2xyxy(box):
	# print(box)
	x,y,w,h= box
	return [x,y,x+w,y+h]

def nms_for_results(result_json, output_json, nms_threshold=0.7):
	all_boxes = json.load(open(result_json, "r"))
	print("Before NMS:", len(all_boxes))
	# reformat
	all_data = {}
	for item in all_boxes:
		imgid = item["image_id"]
		if imgid not in all_data:
			all_data[imgid] = []
		all_data[imgid].append(item)

	num_images = len(all_data)

	after_nms = []
	for i, imgid in enumerate(all_data.keys()): #
		all_items = all_data[imgid]

		all_items.sort(key=lambda x:x["score"], reverse=True)
		pred_boxes = list(map(lambda x:xywh2xyxy(x["bbox"]), all_items))
		cls_scores = list(map(lambda x:x["score"], all_items))

		pred_boxes = Variable(torch.Tensor(pred_boxes))
		cls_scores = Variable(torch.Tensor(cls_scores))

		cls_dets = torch.cat((pred_boxes, cls_scores.unsqueeze(1)), 1)

		keep = nms(pred_boxes, cls_scores, nms_threshold)
		keep = keep.view(-1).long().cpu()

		keep_items = list(map(lambda x:all_items[x], keep))

		after_nms.extend(keep_items)

	print("After NMS:", len(after_nms))
	with open(output_json, "w") as f:
		json.dump(after_nms, f)
	return output_json


def nms_for_results_bbox(all_boxes, output_json, nms_threshold=0.7):
	print("Before NMS:", len(all_boxes))
	# reformat
	all_items = all_boxes

	all_items.sort(key=lambda x:x["score"], reverse=True)
	pred_boxes = list(map(lambda x:xywh2xyxy(x["bbox"]), all_items))
	cls_scores = list(map(lambda x:x["score"], all_items))

	pred_boxes = Variable(torch.Tensor(pred_boxes))
	cls_scores = Variable(torch.Tensor(cls_scores))

	cls_dets = torch.cat((pred_boxes, cls_scores.unsqueeze(1)), 1)

	keep = nms(pred_boxes, cls_scores, nms_threshold)
	keep = keep.view(-1).long().cpu()

	keep_items = list(map(lambda x:all_items[x], keep))
	
	print("After NMS:", len(keep_items))
	with open(output_json, "w") as f:
		json.dump(keep_items, f)
	return keep_items

def main():
	for dataset in ['rico5box']:#,'ricotext','rico','rico2k','rico10k']:
		for split in ['val','test']:
			# result_json = "CenterNet-master/results/output/CenterNet-52/{}-{}/results.json".format(dataset, split)
			result_json = "PyTorch-YOLOv3-master/results/output/{}_{}/{}_{}_results.json".format(dataset,split,dataset,split)
			
			# result_json = '/media/cheer/UI/Project/UIObjectDetection/REMAUI/rico_remaui_text_results.json'
			nms_threshold = 0.7

			output_json = result_json.replace(".json","-nms{}.json".format(nms_threshold))
			nms_for_results(result_json, nms_threshold, output_json)

# main()

# result_json= "/home/cheer/Project/UIObjectDetection/Models/f_box/COMBINED/combined.json"
# result_json = "/home/cheer/Project/UIObjectDetection/Models/faster-rcnn/results/output/res101/ricodefaultAspectRatio_test/detections_test_results.json"
# result_json = "/home/cheer/Project/UIObjectDetection/Models/ensemble/y_combined/combined_inter.json"
# result_json = "/home/cheer/Project/UIObjectDetection/Models/CNN_classifier/mulong/classify/unclassified_data/class_0_pad20_results.json"

if __name__ == '__main__':
	result_json = "/media/cheer/UI/Project/UIObjectDetection/faster_rcnn/output/res101/ricoOriText_test/detections_test_results.json"

	result_json = "/home/cheer/Project/DarkPattern/Code/new_merged_ad_line-addText--test.json"
	nms_threshold = 0.7

	output_json = result_json.replace(".json","-nms{}.json".format(nms_threshold))
	nms_for_results(result_json, output_json, nms_threshold)