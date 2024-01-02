
# UIST2023 Unveiling the Tricks: Automated Detection of Dark Patterns in Mobile Applications

ACM Page: https://dl.acm.org/doi/abs/10.1145/3586183.3606783

***Source code for UIGuard***


Usage: python3 UIGuard.py


UIGuard.py: code for detecting
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
Evaluate the detected dps with the groudtruth dark patterns
Output some metrics(precision, recall, F1)
```

