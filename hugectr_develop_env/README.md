# 开发环境hugectr使用说明

## 文件说明

* data_preprocess:数据预处理部分，包括对数据进行缺失值填充，低频值替换及数据格式转换
* model_train:模型训练部分，包含dcn及deepfm两种模型结构的hugectr训练
* model_inference:模型推理部分，包含dcn及deepfm两种模型的tf推理

## 备注

用户需在开发环境中选择镜像**idockerhub.jd.com/develop_huge/ubantu-cuda10.2-tf2.3-jupyterlab2.1-huge2.2-gpu:1.2**