import os
os.chdir("/notebook/KuAIDemo/hugectr/hugectr_develop/model_train/deepfm")
os.system('./huge_ctr --train ./deepfm.json')