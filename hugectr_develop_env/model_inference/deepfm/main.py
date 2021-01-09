import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import pandas as pd
from dump import DumpToTF,struct
from hugectr_layers import *

NUM_INTEGER_COLUMNS = 13 #计数特征
NUM_CATEGORICAL_COLUMNS = 26 #分类特征
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS #共40列数据

#构建列名（label+I(1-13)+C(1-26)）
def idx2key(idx):
    if idx == 0:
        return 'label'
    return 'I' + str(idx) if idx <= NUM_INTEGER_COLUMNS else 'C' + str(idx - NUM_INTEGER_COLUMNS)

#列名
cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]

#分类取出列名
dense_cols = ['I' + str(i+1) for i in range(NUM_INTEGER_COLUMNS)]
keys_cols = ['C' + str(i+1) for i in range(NUM_CATEGORICAL_COLUMNS)]

def read_a_sample_for_deepfm(args,slot_num):
    """
    从数据集中读取一个样本
    """
    with open(args.dataset,'rb') as file:
        #跳过data-header
        file.seek(4+64+1,0) #默认为0，表示从文件头部开始移动
        #读取一个样本
        length_buffer = file.read(4)
        length = struct.unpack('i', length_buffer)

        label_buffer = file.read(4) # int
        label = struct.unpack('i', label_buffer)[0]

        dense_buffer = file.read(4 * 13) # dense_dim * float
        dense = struct.unpack("13f", dense_buffer)

        keys = []
        for _ in range(slot_num):
            nnz_buffer = file.read(4) # int
            nnz = struct.unpack("i", nnz_buffer)[0]
            key_buffer = file.read(8 * nnz) # nnz * long long 
            key = struct.unpack(str(nnz) + "q", key_buffer)
            keys += list(key)

        check_bit_buffer = file.read(1) # char
        check_bit = struct.unpack("c", check_bit_buffer)[0]

    label = np.int64(label)
    dense = np.reshape(np.array(dense, dtype=np.float32), newshape=(1, 13))
    keys = np.reshape(np.array(keys, dtype=np.int64), newshape=(1, 26, 1))

    return label, dense, keys

def read_1000_sample_for_deepfm(args,slot_num):
    """
    从数据集中读取一个样本
    """
    label_batch = []
    dense_batch = []
    keys_batch = []
    
    with open(args.dataset,'rb') as file:
        for i in range(1000):
            #跳过data-header
            file.seek(4+64+1,0) #默认为0，表示从文件头部开始移动
            #读取一个样本
            length_buffer = file.read(4)
            length = struct.unpack('i', length_buffer)

            label_buffer = file.read(4) # int
            label = struct.unpack('i', label_buffer)[0]

            dense_buffer = file.read(4 * 13) # dense_dim * float
            dense = struct.unpack("13f", dense_buffer)

            keys = []
            for _ in range(slot_num):
                nnz_buffer = file.read(4) # int
                nnz = struct.unpack("i", nnz_buffer)[0]
                key_buffer = file.read(8 * nnz) # nnz * long long 
                key = struct.unpack(str(nnz) + "q", key_buffer)
                keys += list(key)

            check_bit_buffer = file.read(1) # char
            check_bit = struct.unpack("c", check_bit_buffer)[0]
            
            label_batch.append(label)
            dense_batch.append(dense)
            keys_batch.append(keys)
            
#     label = np.int64(label_batch)
    label = np.reshape(np.array(label_batch,dtype = np.int64),newshape = (1000,1))
    dense = np.reshape(np.array(dense_batch, dtype=np.float32), newshape=(1000, 13))
    keys = np.reshape(np.array(keys_batch, dtype=np.int64), newshape=(1000, 26, 1))

    return label, dense, keys

#读取数据（np读取csv）
def read_1000_sample_for_deepfm_np(args):  
    data = pd.read_csv(args.dataset,sep = ' ',names = cols)
    
    label_batch = data['label']
    dense_batch = data[dense_cols]
    keys_batch = data[keys_cols]
    
    label = np.reshape(np.array(label_batch,dtype = np.int64),newshape = (1000,1))
    dense = np.reshape(np.array(dense_batch, dtype=np.float32), newshape=(1000, 13))
    keys = np.reshape(np.array(keys_batch, dtype=np.int64), newshape=(1000, 26, 1))
    return label,dense,keys
        
def deepfm_model(args):
    """
    构造计算图，并使用训练好的模型进行权重初始化
    """
    #定义超参数
    batchsize = 1000
    slot_num = 26
    max_nnz_per_slot = 1
    
    dense_dim = 13
    emb_size = 10
    #模型配置文件路径
    samples_dir = r'../../deepfm/'
    model_json = os.path.join(samples_dir,r'deepfm.json')
    
    #定义训练好的模型文件
    sparse_model_names = args.sparse_models
    dense_model_name = args.dense_model
    
    #初始化权重
    dump = DumpToTF(sparse_model_names = sparse_model_names,dense_model_name = dense_model_name,model_json = model_json,non_training_params_json = None)
    
    #定义输出checkpoint
    checkpoint_path = r'./tf_checkpoint/deepfm/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = os.path.join(checkpoint_path,"deepfm_model")
    
    #---构造计算图---#
    #清空计算图信息
    tf.reset_default_graph() 
    graph = tf.Graph()
    with graph.as_default():
        #dense_input
        dense_input = tf.placeholder(shape = (batchsize,dense_dim),dtype = tf.float32,name = 'dense_input')
        #sparse_input
        sparse_input = tf.placeholder(shape = (batchsize,slot_num,max_nnz_per_slot),dtype = tf.int64,name = 'sparse_input')\
        # labels
        labels = tf.placeholder(shape=(batchsize,1), dtype=tf.float32, name='labels')
        #dump embedding to tf
        layer_name, init_values = dump.parse_embedding().__next__()
        vocabulary_size = init_values.shape[0]
        embedding_feature = embedding_layer(sparse_input, init_values, combiner=0)  #(,26,10)
        #slice1
        slice11 = tf.slice(embedding_feature, [0, 0, 0], [-1, -1, emb_size])   #(,26,10)
        slice12 = tf.slice(embedding_feature, [0, 0, emb_size], [-1, -1, 1])  #(,26,1)
        #multiply1
        layer_type = "Multiply"
        layer_bytes = dense_dim * emb_size * 4
        weight1 = dump.parse_dense(layer_bytes, layer_type, weights_dims=[dense_dim, emb_size])  #(13,10)
        multiply1 = multiply_layer(dense_input, weight1)  # (,13,10)
        #multiply2
        layer_type = "Multiply"
        layer_bytes = dense_dim * 1 * 4
        weight2 = dump.parse_dense(layer_bytes, layer_type, weights_dims=[dense_dim, 1]) 
        multiply2 = multiply_layer(dense_input,weight2)  #(,13,1)
        #concat1
        concat1 = tf.concat([slice11,multiply1],1) #(,39,10)
        #resahpe1
        reshape1 = tf.reshape(concat1,[-1,(slot_num+dense_dim)*emb_size]) #(,390)
        
        #---DNN_part---
        #fc1
        layer_type = "InnerProduct"
        num_output = 400
        layer_bytes = (reshape1.shape[1] * num_output + 1 * num_output) * 4
        weight_fc1, bias_fc1 = dump.parse_dense(layer_bytes, layer_type,
                                                in_feature=reshape1.shape[1],
                                                out_feature=num_output)
        fc1 = innerproduct_layer(reshape1, weight_fc1, bias_fc1)
        #relu1
        relu1 = tf.nn.relu(fc1)
        #fc2
        layer_type = "InnerProduct"
        num_output = 400
        layer_bytes = (relu1.shape[1] * num_output + 1 * num_output) * 4
        weight_fc2, bias_fc2 = dump.parse_dense(layer_bytes, layer_type,
                                                in_feature=relu1.shape[1],
                                                out_feature=num_output)
        fc2 = innerproduct_layer(relu1, weight_fc2, bias_fc2)
        #relu2
        relu2 = tf.nn.relu(fc2)
        #fc3
        layer_type = "InnerProduct"
        num_output = 400
        layer_bytes = (relu2.shape[1] * num_output + 1 * num_output) * 4
        weight_fc3, bias_fc3 = dump.parse_dense(layer_bytes, layer_type,
                                                in_feature=relu2.shape[1],
                                                out_feature=num_output)
        fc3 = innerproduct_layer(relu2, weight_fc3, bias_fc3)
        #relu3
        relu3 = tf.nn.relu(fc3)
        #fc4
        layer_type = "InnerProduct"
        num_output = 1
        layer_bytes = (relu3.shape[1] * num_output + 1 * num_output) * 4
        weight_fc4, bias_fc4 = dump.parse_dense(layer_bytes, layer_type,
                                                in_feature=relu3.shape[1],
                                                out_feature=num_output)  
        fc4 = innerproduct_layer(relu3,weight_fc4,bias_fc4)  #(,1)
        
        #---fm2---
        #fmorder2
        sumed_square = tf.square(tf.reduce_sum(concat1, axis=1))  #(,10)
        squared_sum = tf.reduce_sum(tf.square(concat1), axis=1)   #(,10)
        #reducesum1
        reducesum1 = 0.5 * tf.reduce_sum(tf.subtract(sumed_square, squared_sum), axis=1,keep_dims = True)  #(,1)
        
        #---fm1---
        #reshape2
        reshape2 = tf.squeeze(slice12,axis = -1) #(,26)
        #concat2
        multiply2 = tf.squeeze(multiply2,axis = -1)
        concat2 = tf.concat([reshape2,multiply2],1) #(,39)
        #reducesum2
        reducesum2 = tf.reduce_sum(concat2,1,keep_dims = True) #(,1)
        
        #add
        add = tf.add(tf.add(fc4,reducesum1),reducesum2) #(,1)      
        output = tf.nn.sigmoid(add)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=add, labels=labels))
        
        #检查是否所有dense层权重都被解析
        dump.read_dense_complete()

        init_op = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        saver = tf.train.Saver()
    
    with tf.Session(graph = graph) as sess:
        sess.graph.finalize()
        sess.run(init_op)
        #检查推理输出
        label,dense,keys = read_1000_sample_for_deepfm_np(args)
        keys[keys== -1] = vocabulary_size
        output = sess.run(loss,feed_dict={dense_input: dense,
                                          sparse_input: keys,
                                         labels:label})
        print("[INFO] output = %f" %output)
        
        #保存checkpoint文件
        saver.save(sess,checkpoint,global_step = 0)
        print("[INFO] save done")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #添加命令行参数
    parser.add_argument("dataset",type = str,help = "where to find criteo dataset")
    parser.add_argument("dense_model",type = str,help = "where to find dense model file")
    parser.add_argument("sparse_models",nargs = "+", type = str,help = "where to find sparse model files")
    
    #命令行参数转对象
    args = parser.parse_args()
    
    print("[INFO] running deepfm...")
    deepfm_model(args)
    