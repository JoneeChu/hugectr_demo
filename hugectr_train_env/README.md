# 训练环境hugectr使用说明

## 前置条件

用户已在开发环境中成功进行数据预处理和模型训练

## 操作流程

1. 开发环境中创建python脚本，其中执行能成功训练hugectr的shell命令（demo中为/notebook/KuAIDemo/hugectr/hugectr_develop/model_train/dcn/hugectr_train.py）
2. 在kuai模型训练一栏中创建作业
3. 启动文件选择开发环境中对应的python脚本
4. 软件环境一栏中，镜像选择其他仓库镜像，名称填**idockerhub.jd.com/train_hugectr/tf2.4-huge2.2-gpu:latest**
5. 硬件资源一栏中，资源池选择模型**训练LAMBDA资源**
6. 填好其他必填项后点击保存并启动