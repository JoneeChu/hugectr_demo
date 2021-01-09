#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from traceback import print_exc

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp

NUM_INTEGER_COLUMNS = 13 #计数特征
NUM_CATEGORICAL_COLUMNS = 26 #分类特征
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS #共40列数据

INT_NAN_VALUE = np.iinfo(np.int32).min #整型值填充
CAT_NAN_VALUE = '80000000' #分类值填充


# In[2]:


from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession,HiveContext
import pyspark.sql.functions as f
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark import SparkFiles
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession,HiveContext
# import add_v2 #计算函数 
sparkConf = SparkConf()

# 设置Driver进程的内存
sparkConf.set('spark.driver.memory', '16G')
# 设置Driver的CPU core数量
sparkConf.set('spark.driver.cores', '8')
# 设置Spark作业总共要用多少个Executor进程来执行
sparkConf.set("spark.executor.instances", "200")
sparkConf.set("spark.dynamicAllocation.maxExecutors","2000")
# 设置每个Executor进程的CPU core数量
sparkConf.set("spark.executor.cores", "3")
# 设置每个Executor进程的内存
sparkConf.set("spark.executor.memory", "28G")
# 设置Spark应用的名称
sparkConf.set("spark.app.name", "data_processing_1")
sparkConf.set("spark.shuffle.service.enabled", "true")
sparkConf.set("spark.dynamicAllocation.enabled", "true")
sparkConf.set("spark.dynamicAllocation.executorIdleTimeout", "600s")
sparkConf.set("spark.driver.maxResultSize", "10g")
# 打开pyarrow
sparkConf.set("spark.sql.execution.arrow.enabled", "true")
sparkConf.set("spark.sql.shuffle.partitions", "10000")

spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
sc = spark.sparkContext


# In[3]:


#构建列名（label+I(1-13)+C(1-26)）
def idx2key(idx):
    if idx == 0:
        return 'label'
    return 'I' + str(idx) if idx <= NUM_INTEGER_COLUMNS else 'C' + str(idx - NUM_INTEGER_COLUMNS)

#填充缺失值
def _fill_missing_features_and_split(chunk,series_list_dict):
    for cid,col in enumerate(chunk.columns):
        NAN_VALUE = INT_NAN_VALUE if cid <= NUM_INTEGER_COLUMNS else CAT_NAN_VALUE
        result_series = chunk[col].fillna(NAN_VALUE)
        series_list_dict[col].append(result_series)

#更改出现次数少于6的值，对数据进行归一化
def _merge_and_transform_series(src_series_list, col, dense_cols,
                                normalize_dense):
    result_series = pd.concat(src_series_list)

    if col != 'label':
        unique_value_counts = result_series.value_counts()
        unique_value_counts = unique_value_counts.loc[unique_value_counts >= 6]
        unique_value_counts = set(unique_value_counts.index.values)
        NAN_VALUE = INT_NAN_VALUE if col.startswith('I') else CAT_NAN_VALUE
        result_series = result_series.apply(
                lambda x: x if x in unique_value_counts else NAN_VALUE)

    if col == 'label' or col in dense_cols:
#         result_series = result_series.astype(np.float64)
        result_series = result_series.astype(np.int64)
        le = skp.LabelEncoder()
        result_series = pd.DataFrame(le.fit_transform(result_series))
        if col != 'label':
           result_series = result_series + 1
    else:
        oe = skp.OrdinalEncoder(dtype=np.int64)
        result_series = pd.DataFrame(oe.fit_transform(pd.DataFrame(result_series)))
        result_series = result_series + 1


    if normalize_dense != 0:
        if col in dense_cols:
           mms = skp.MinMaxScaler(feature_range=(0,1))
           result_series = pd.DataFrame(mms.fit_transform(result_series))

    result_series.columns = [col]

    min_max = (np.int64(result_series[col].min()), np.int64(result_series[col].max()))
#     if col != 'label':
#         logging.info('column {} [{}, {}]'.format(col, str(min_max[0]),str(min_max[1])))

    return [result_series, min_max]

#数据转dataframe，是否做特征交叉
def _merge_columns_and_feature_cross(series_list, min_max, feature_pairs,
                                     feature_cross):
    name_to_series = dict()
    for series in series_list:
        name_to_series[series.columns[0]] = series.iloc[:,0]
    df = pd.DataFrame(name_to_series)
    cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]
    df = df.reindex(columns=cols)

    offset = np.int64(0)
    for col in cols:
        if col != 'label' and col.startswith('I') == False:
            df[col] += offset
#             logging.info('column {} offset {}'.format(col, str(offset)))
            offset += min_max[col][1]

    if feature_cross != 0:
        for idx, pair in enumerate(feature_pairs):
            col0 = pair[0]
            col1 = pair[1]

            col1_width = int(min_max[col1][1] - min_max[col1][0] + 1)

            crossed_column_series = df[col0] * col1_width + df[col1]
            oe = skp.OrdinalEncoder(dtype=np.int64)
            crossed_column_series = pd.DataFrame(oe.fit_transform(pd.DataFrame(crossed_column_series)))
            crossed_column_series = crossed_column_series + 1

            crossed_column = col0 + '_' + col1
            df.insert(NUM_INTEGER_COLUMNS + 1 + idx, crossed_column, crossed_column_series)
            crossed_column_max_val = np.int64(df[crossed_column].max())
#             logging.info('column {} [{}, {}]'.format(
#                 crossed_column,
#                 str(df[crossed_column].min()),
#                 str(crossed_column_max_val)))
            df[crossed_column] += offset
#             logging.info('column {} offset {}'.format(crossed_column, str(offset)))
            offset += crossed_column_max_val

    return df


# In[4]:


#数据预处理
def preprocess(df, normalize_dense, feature_cross):
    cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)] #构造列名
    series_list_dict = dict() #一个字典，键为列名，值为每列数据
    
    if "grouper" in set(df.columns): #去除"grouper"列
        df = df.drop("grouper",axis=1)
    df = df.apply(pd.to_numeric,errors = "ignore") #将string转换为数字
    df.columns = cols #修改列名
    
#     logging.info('read a CSV file')
#     df = pd.read_csv(src_txt_name,sep='\t',names=cols,chunksize=131072) #构造一个TextFileReader
#     df = pd.read_csv(src_txt_name,sep='\t',names=cols) #直接构造一个dataframe
     
#     logging.info('_fill_missing_features_and_split')
    for col in cols:
        series_list_dict[col] = list()
    _fill_missing_features_and_split(df,series_list_dict)
    
#     logging.info('_merge_and_transform_series')
    futures = list()
    dense_cols = [idx2key(idx+1) for idx in range(NUM_INTEGER_COLUMNS)] #保存整型列名
    dst_series_list = list() #存储每列值
    min_max = dict() #存储每列元素最小最大值
    for col,src_series_list in series_list_dict.items():
        future = _merge_and_transform_series(src_series_list,col,dense_cols,normalize_dense) #每个future都是每列数据在填充好值后进行数据归一化后的结果（结果是一个包含两个元素的列表，第一个元素是一个dataframe，第二个元素是一个min_max元组）
        futures.append(future)
    
    #列中元素值与每列最小最大值分开存储
    for future in futures:
        col = None
        for idx, ret in enumerate(future):
            try:
                if idx == 0:
                    col = ret.columns[0]
                    dst_series_list.append(ret)
                else:
                    min_max[col] = ret
            except:
                print_exc()
    
#     logging.info('_merge_columns_and_feature_cross')
    feature_pairs = [('C1', 'C2'), ('C3', 'C4')]
    df = _merge_columns_and_feature_cross(dst_series_list, min_max, feature_pairs,
                                              feature_cross)
    return df
#     logging.info('write to a CSV file')
#     df.to_csv(dst_txt_name, sep=' ', header=False, index=False)

#     logging.info('done!')


# In[5]:


cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]


# In[6]:


types = "label int,"
for i in range(1,14):
    types += cols[i]
    types += " "
    types += "float"
    types += ","
for i in range(14,39):
    types += cols[i]
    types += " "
    types += "int"
    types += ","
types = types + "C26 int"


# In[7]:


types


# In[8]:


type_all = types #要生成的结果的数据类型
@pandas_udf(type_all, PandasUDFType.GROUPED_MAP) #签名，将spark dataframe转换成pandas dataframe
def pre_udf(pdf):
    rest=preprocess(pdf,1,0)
    return rest


# In[9]:


# df = spark.read.csv("input/train_new.shuf.txt",sep = '\t') #小数据
df = spark.read.csv("input/train.txt",sep = "\t") #大数据


# In[30]:


df.show(2)


# In[22]:


df.printSchema()


# In[10]:


nparts = 100
output = df.select('*',(f.ceil(f.rand()*nparts)).alias('grouper')).groupby("grouper").apply(pre_udf)


# In[11]:


output


# In[12]:


output.show(2)


# In[13]:


output.write.format('csv').option('header','false').save('input/train_big_new_out_1.csv')


# In[ ]:




