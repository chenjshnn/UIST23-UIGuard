import os, json, cv2, time
from glob import glob
from nms import nms_for_results_bbox

from elementDetection.FASTER_RCNN.detect_elements import test_single
from elementDetection.get_ocr import detect_text_paddle
from elementDetection.merge_dets_single import merge_dets_single

print("==> Load detection Model")
from gather_basic_info import get_color_status_icon
print("==> Finished Importing")
from template_matching.template_matching import get_ad_icons
from merge_tm_checkgroup import merge_tm_results_checkgroup

from rule_check import predict_type

from tqdm import tqdm

print("==> Finished Importing")

class UIGuard:
	dpCode2dpRealName = {"II-AM-G-SMALL": "Interface Inference", 
						 "FA-G-WATCHAD": "Forced Action",  
						 "SN-FC": "Forced Continuity", 
						 "II-AM-TWE": "Aesthetic Manipulation", 
						 "II-AM-DA": "Disguised Ad", 
						 "FA-SOCIALPYRAMID": "Social Pyramid", 
						 "FA-GAMIFICATION": "Forced Action", 
						 "NG-RATE": "Nag to rate", 
						 "NG-UPGRADE": "Nag to upgrade",
						 "FA-G-COUNTDOWNAD": "Forced Action", 
						 "II-AM-FH": "False Hierarchy",
						 "FA-G-PRO": "Forced Action", 
						 "II-PRE-FOLLOW": "Preselection", 
						 "II-PRE-NOTIFICATION": "Preselection", 
						 "II-PRE-PRIVACY": "Preselection", 
						 "II-PRE-USAGE-DATA": "Preselection",
						 "II-PRE": "Preselection",
						 "FA-Privacy": "Forced Action",

						 }

	def extract_property(self, image_path, output_root, img_cv, vis=False):
		## get results from faster rcnn
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_1_det_frcnn.json")
		print(tmp_path)
		start_time = time.time()
		fasterRCNN_detection  = test_single(image_path, tmp_path, vis=vis)
		# output: {imgid: [{"category": Button, "bbox":(x1,y1,x2,y2), "score":float}]}
		fasterRCNN_detection = list(fasterRCNN_detection.values())[0]
		# output: [{"category": Button, "bbox":(x1,y1,x2,y2), "score":float},...]
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_2_det_frcnn_nms.json")
		fasterRCNN_detection_nms = nms_for_results_bbox(fasterRCNN_detection, tmp_path)
		# output: {imgid: [{"category": Button, "bbox":(x1,y1,x2,y2), "score":float}]}
		print("Non-Text Detection Using {:.02f}s".format(time.time() - start_time))


		## get ocr results
		start_time = time.time()
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_3_det_ocr.json")
		ocr_detection = detect_text_paddle(img_cv, tmp_path, vis=vis)
		## output: [{"category": pText, "bbox":[x1,y1,x2,y2], "text": str, "score": float}]
		print("Text Detection Using {:.02f}s".format(time.time() - start_time))


		## merge both detections
		start_time = time.time()
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_4_det_merged.json")
		merged_dets = merge_dets_single(fasterRCNN_detection_nms, ocr_detection, img_cv, tmp_path, vis=vis)
		# output: ["catgory": Button, pText, "text":str, "bbox":[x1,y1,x2,y2], "score": float, "match": bool, 
		# 		   "text_items": [{"category": pText, "bbox":[x1,y1,x2,y2], "text": str, "score": float}]]

		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_5_det_merged_nms.json")
		merged_dets_nms = nms_for_results_bbox(merged_dets, tmp_path)
		print("Merging Detection Using {:.02f}s".format(time.time() - start_time))


		# ------
		### get colors, checkbox, icon semantic results

		start_time = time.time()
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_6_gather_info.json")
		gather_info = get_color_status_icon(merged_dets_nms, image_path, tmp_path)
		print("Getting color and status Using {:.02f}s".format(time.time() - start_time))


		### get ad icons

		start_time = time.time()
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_7_ad_icons.json")
		ad_icons_close, ad_icons_info = get_ad_icons(img_cv, tmp_path, vis=vis)
		print("Getting ad icons TM Using {:.02f}s".format(time.time() - start_time))


		## add ad icons to gather info
		## merge checkbox with its text and make them a check_group
		start_time = time.time()
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_8_merged.json")
		all_properties = merge_tm_results_checkgroup(gather_info, ad_icons_close, ad_icons_info, tmp_path)
		print("Merging checkbox group Using {:.02f}s".format(time.time() - start_time))


		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_all_properties.json")
		with open(tmp_path, "w") as f:
			json.dump(all_properties, f)

		return all_properties

	def darkpatternChecker(self, all_properties, img_cv, image_path, output_root, vis=False):
		img_h, img_w, _ = img_cv.shape
		output_vis_root = os.path.join(output_root, "dp_results")
		if not os.path.exists(output_vis_root):
			os.makedirs(output_vis_root)
		final_results = predict_type(all_properties, img_h, img_w, output_vis_root, image_path, vis=vis)
		## output: {dp_type: [{"label": dp_type, "bbox":[x1,y1,x2,y2], "subType": desc,
		##			 		   "flag": str, "text": str, "fg_lum": float, "bg_lum":float,
		##					   "con":float, "bg_color", [r,g,b], "fg_color":[r,g,b],
		# 					   "children": [{"category":Button, "bbox":[x1,y1,x2,y2], "score": float, "matched": bool, "text_items": [...]}]}]}
		tmp_path = os.path.join(output_root, os.path.basename(image_path).split(".")[0]+"_dp_checker.json")
		with open(tmp_path, "w") as f:
			json.dump(final_results, f)
		return final_results

	def UIGuard(self, image_path, output_root, vis):
		img_cv = cv2.imread(image_path)
		all_properties = self.extract_property(image_path, output_root, img_cv, vis)


		start_time = time.time()
		final_results = self.darkpatternChecker(all_properties, img_cv, image_path, output_root, vis=True)
		print("Rule Checking Using {:.02f}s".format(time.time() - start_time))


		start_time = time.time()
		android_output = self.organise_output_for_android(final_results)
		print("Reorganise results Using {:.02f}s".format(time.time() - start_time))
		print("++ android_output", android_output)

		return android_output


	def organise_output_for_android(self, output):
		android_output = {"results":[]}
		for each_dp_type, item_list in output.items():
			for item in item_list:
				typ_ = self.dpCode2dpRealName.get(each_dp_type, each_dp_type)
				tmp_object = {"type": typ_,
							  "desc": item["subType"].split("]")[1],
							  "text_content": item.get("text", ""),
							  "container_bbox": item["bbox"],
							  "children": []}
				for child in item.get("children", []):
					tmp_object["children"].append({"bbox": child["bbox"]})
				android_output["results"].append(tmp_object)
		return android_output



if __name__ == '__main__':
	# main()
	uiguard = UIGuard()


	test_data_root = "test_data"
	all_test_data = glob(test_data_root + "/**/**.jpg", recursive=True)
	# test_data_root = "/Users/che444/Desktop/DPCODE-CLEAN/finalCode/data/annotations/Rico_testset/testset"
	# all_test_data = glob(test_data_root + "/**.jpg")

	output_root = os.path.join(test_data_root, "detection")
	if not os.path.exists(output_root):
		os.makedirs(output_root)

	start_time = time.time()
	for image_path in tqdm(all_test_data):
		# if "20177" not in image_path:
		# 	continue
		uiguard.UIGuard(image_path, output_root, vis=False)
	end_time = time.time()
	print(f"Using {(end_time-start_time)/len(all_test_data)}/img")









