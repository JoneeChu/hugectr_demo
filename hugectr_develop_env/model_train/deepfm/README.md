# hugectr deepfm模型训练使用说明

## 文件说明

* criteo:hugectr格式训练数据集
* criteo_test:hugectr格式测试数据集
* criteo2hugectr:将原预处理后数据转换为hugectr格式数据的执行文件
* dcn.json:deepfm模型配置文件
* file_list.tmp:转hugectr格式临时文件
* file_list.txt:训练数据转hugectr文件列表
* file_list_test.txt:测试数据转hugectr文件列表
* huge_ctr:进行模型训练的执行文件

## 操作流程

1. 直接执行./huge_ctr --train ./deepfm.json