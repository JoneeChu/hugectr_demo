# 数据预处理使用说明

## 文件说明

本文件夹下的三种进行数据预处理的方式任选其一，都能达到预处理需求，具体不同点如下：

* preprocess_multithreading.py:原项目中使用的方式，借助多线程与多进程并行数据处理，提高处理速度
* preprocess_single.py:根据多线程脚本进行还原，是使用spark进行分布式转换与使用多线程进行并行转换的中间状态，更利于理清预处理流程
* preprocess_pandas_udf.py:借助单机脚本使用spark pandas_udf进行分布式处理

其他文件说明

preprocess.sh:该文件包含如下三个操作

* 原始数据的解压缩

* 对训练数据集进行shuffle

* 对训练数据集进行预处理，选择脚本为上面介绍的三个文件中的一个

  

## 操作流程

1. 进入当前操作目录:cd KuAIDemo/hugectr/hugectr_develop/data_preprocess/

2. 执行脚本preprocess.sh:bash preprocess dcn 1 0（此时会在当前目录生成dcn_data文件夹，文件夹中包含test、train、train.out.txt、val及valtest共五个文件）

3. 将预处理好的数据转为hugectr格式数据（鉴于训练时最好让数据同结构文件保持同目录，此处需先进入开发环境中的训练目录）

   1. 先进入训练目录:cd ../model_train/dcn

   2. 当前目录下进行数据转换后将训练集写入criteo目录，测试集写入criteo_test目录:

      1. /criteo2hugectr ../../data_preprocess/dcn_data/train criteo/sparse_embedding file_list.txt  
      2. ./criteo2hugectr ../../data_preprocess/dcn_data/val criteo_test/sparse_embedding file_list_test.txt

## 备注

原始数据集可到我分享的[鲸盘链接](http://pan.jd.com/sharedInfo/3EFCA673B09A189EEEC946AFBC5707A7)下载



