
# Unveiling the Tricks: Automated Detection of Dark Patterns in Mobile Applications

**Accepted to UIST2023**

## RESOURCE
- Paper: [ACM](https://dl.acm.org/doi/abs/10.1145/3586183.3606783)
- Video: [Youtube](https://www.youtube.com/watch?v=PkXHuPkatpk&t=16167s)
- Dataset: [Zenodo](https://zenodo.org/records/8126443)
- Code: [src](src/)
- FRCNN Pretrained Models: [Zenodo](https://zenodo.org/record/8098605)

## CODE
All code is tested under Ubuntu 16.04, Cuda 9.0, PyThon 3.9, torch 1.12.1, Nvidia 1080 Ti and also tested under MacOS 13.2.1, Apple M1 Pro



## Usage 

Usage: python3 UIGuard.py


UIGuard.py: code for detecting deceptive patterns
```
Modify L164 test_data_root to your data stored path
Modify L177 parameter _vis_ to decide whether draw the result or not
Update the path to the classification models in iconModel/get_iconLabel.py and statusModel/get_status.py
Update the path to the templates in template_matching/template_matching.py
```


rule_check.py: examine the dark patterns existence based on the extracted properties
```
L29-L32
flag_icon = True
flag_TM = True
flag_status = True
flag_grouping = True

Modify them to choose whether to use icon information (flag_icon) for examination.
Similar to other flags
```

evaluate.py
```
Evaluate the detected dps against the groudtruth dark patterns
Output some metrics(precision, recall, F1)
```

If you have any configuration problems with Faster RCNN, please refer to https://github.com/chenjshnn/Object-Detection-for-Graphical-User-Interface.
