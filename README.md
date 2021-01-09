# HugeCTR_demo使用说明 #

## HugeCTR介绍 ##

HugeCTR是一个设计用于CTR任务的高效GPU训练框架，其主要具有以下三个优点：

* 速度快：较平常单机版本CPU中CTR任务训练速度快约50倍，这主要得益于GPU哈希及动态插入、多节点训练及超大embedding支持与混淆精度训练
* 全方位：从数据预处理到模型训练的整个流程都有较为详细的介绍
* 易上手：不论你是什么职业，是否具有一定经验，都可以迅速使用本框架

想细致了解该框架请移步[官方网站](https://github.com/NVIDIA/HugeCTR/tree/v2.1)

## demo项目说明

本项目主要是在原github的基础上，重新梳理数据预处理、模型训练流程并增加模型推理内容，主要内容如下：

* 数据预处理：对原始数据进行空值及低频次内容填充，对稀疏值进行哈希，对最终处理好的数据进行hugectr格式转换
* 模型训练：根据dcn或deepfm网络结构书写模型配置文件
* 模型推理：根据训练好的模型参数初始化模型结构，使用Tensorflow重写模型结构 

## demo文件说明 ##

* hugectr_develop_env:开发环境中使用hugectr，包括数据预处理、模型训练及模型推理
* hugectr_train_env:训练环境中使用hugectr，创建hugectr模型训练作业

## 用户使用说明 ##

用户直接测试所给demo或基于自身业务进行相关测试时可遵循以下使用步骤：

1. 数据预处理：对数据进行缺失值填充、低频次值替换及归一化等操作，预处理完成后需对数据进行hugectr格式转换（本demo中数据来源自[kaggle](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)，用户可使用其他数据进行测试）
2. 模型训练：用户可在开发环境或训练环境中进行模型训练（指定镜像中）
3. 模型推理：基于已经训练好的模型参数进行dump to tensorflow操作

## 备注

原始数据集可到我分享的[鲸盘链接](http://pan.jd.com/sharedInfo/3EFCA673B09A189EEEC946AFBC5707A7)下载



