# hugectr deepfm模型推理使用说明

## 文件说明

1. dump.py:将训练得到的model文件进行解析得到初始参数
2. hugectr_layers.py:使用tf重写模型结构时所需具体操作
3. main.py:主脚本
4. tf_checkpoint:执行主脚本得到的ckpt文件
5. test.txt:从原始数据集截取的千行预处理后的样本（csv格式）

## 执行步骤

1. 数据预处理后获取转换后的hugectr格式数据（不用hugectr格式也可以，直接使用预处理后的csv文件）
2. 模型训练获取model文件
3. 对模型配置文件deepfm.json进行修改（更改迭代次数、去除dropout等）
4. 执行主脚本生成ckpt文件用于之后的模型推理、部署等操作（python3 main.py dataset dense_model sparse_model0 sparse_model1 ...）

