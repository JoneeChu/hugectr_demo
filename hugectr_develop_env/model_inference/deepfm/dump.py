import struct
import os 
import numpy as np
import json

class DumpToTF(object):
    def __init__(self,sparse_model_names,dense_model_name,model_json,non_training_params_json = None):
        self.sparse_model_names = sparse_model_names #字符串列表
        self.dense_model_name = dense_model_name #字符串
        self.model_json = model_json #整个模型的json文件
        self.non_training_params_json = non_training_params_json #非训练参数
        
        self.model_content = None
        self.embedding_layers = None
        self.dense_layers = None
        
        self.parse_json()
        self.offset = 0
        
    def parse_json(self):
        """
        解析模型json文件获取整个模型的层信息，保存在列表中
        解析非训练参数json文件，获取非训练参数
        returns:
        [embedding_layer,dense_layer0,dense_layer1,...],
        [non-training-params]
        """
        print("[INFO] begin to parse model json file:%s"%self.model_json)
        try:
            with open(self.model_json,'r') as model_json:
                self.model_content = json.load(model_json)
                layers = self.model_content["layers"]
                #embedding_layers
                self.embedding_layers = []
                for index in range(1,len(layers)): #第一层用于保存数据信息
                    if layers[index]["type"] not in ("DistributedSlotSparseEmbeddingHash",
                                                    "LocalizedSlotSparseEmbeddingHash"):
                        break
                    else:
                        self.embedding_layers.append(layers[index])
                
                #dense_layers
                self.dense_layers = layers[1+len(self.embedding_layers):]

        except BaseException as error:
            print(error)
    def parse_embedding(self):
        """
        一次获取一层embedding表
        """
        #先判断是否已经解析模型json文件（获取模型基本结构）
        if self.model_content is None:
            self.parse_json()
        
        for index,layer in enumerate(self.embedding_layers):
            print("[INFO] begin to parse embedding weights:%s"%layer["name"])
            
            each_key_size = 0
            layer_type = layer["type"]
            embedding_vec_size = layer["sparse_embedding_hparam"]["embedding_vec_size"]
            vocabulary_size = layer["sparse_embedding_hparam"]["vocabulary_size"]
            
            if layer_type == "DistributedSlotSparseEmbeddingHash":
                # sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size
                each_key_size = 8 + 4 * embedding_vec_size
            elif layer_type == "LocalizedSlotSparseEmbeddingHash":
                # sizeof(TypeHashKey) + sizeof(TypeHashValueIndex) + sizeof(float) * embedding_vec_size
                each_key_size = 8 + 8 + 4 * embedding_vec_size
            
            embedding_table = np.zeros(shape = (vocabulary_size,embedding_vec_size),dtype= np.float32)
            
            #用训练好的模型参数初始化权重
            with open(self.sparse_model_names[index],'rb') as file: #读取二进制文件（'rb'）
                try:
                    while True:
                        buffer = file.read(each_key_size)
                        if len(buffer) == 0:
                            break
                        if layer_type == "DistributedSlotSparseEmbeddingHash":
                            key = struct.unpack("q", buffer[0:8])
                            values = struct.unpack(str(embedding_vec_size) + "f", buffer[8:])
                        elif layer_type == "LocalizedSlotSparseEmbeddingHash":
                            key, slot_id = struct.unpack("2q", buffer[0:11])
                            values = struct.unpack(str(embedding_vec_size) + "f", buffer[11:])
                        embedding_table[key] = values
                except BaseException() as error:
                    print(error)
            yield layer["name"],embedding_table
    
    def parse_dense(self,layer_bytes,layer_type,**kwargs): 
        """
        一次获取一层权重
        """
        #是否解析模型json文件
        if self.model_content is None:
            self.parse_json()
            self.offset = 0 #定义偏移量，用于判断是否全部读取完模型文件
        with open(self.dense_model_name, 'rb') as file:
            print("[INFO] begin to parse dense weights: %s" %layer_type)

            file.seek(self.offset, 0)

            buffer = file.read(layer_bytes)

            if layer_type == "BatchNorm":
                # TODO
                pass
            elif layer_type == "InnerProduct":
                in_feature = kwargs["in_feature"]
                out_feature = kwargs["out_feature"]

                weight = struct.unpack(str(in_feature * out_feature) + "f", buffer[ : in_feature * out_feature * 4])
                bias = struct.unpack(str(out_feature) + "f", buffer[in_feature * out_feature * 4 : ])

                weight = np.reshape(np.float32(weight), newshape=(in_feature, out_feature))
                bias = np.reshape(np.float32(bias), newshape=(1, out_feature))

                self.offset += layer_bytes
                return weight, bias


            elif layer_type == "MultiCross":
                vec_length = kwargs["vec_length"]
                num_layers = kwargs["num_layers"]

                weights = []
                biases = []

                each_layer_bytes = layer_bytes // num_layers

                for i in range(num_layers):
                    weight = struct.unpack(str(vec_length) + "f", buffer[i*each_layer_bytes : i*each_layer_bytes + vec_length * 4])
                    bias = struct.unpack(str(vec_length) + "f", buffer[i*each_layer_bytes + vec_length * 4 : (i+1)*each_layer_bytes])

                    weights.append(np.reshape(np.float32(weight), newshape=(1, len(weight))))
                    biases.append(np.reshape(np.float32(bias), newshape=(1, len(bias))))

                self.offset += layer_bytes

                return weights, biases

            elif layer_type == "Multiply":
                weights_dims = kwargs["weights_dims"]

                weight = struct.unpack(str(weights_dims[0] * weights_dims[1]) + "f",
                                   buffer[ : weights_dims[0] * weights_dims[1] * 4])

                weight = np.reshape(np.float32(weight), newshape=(weights_dims[0], weights_dims[1]))

                self.offset += layer_bytes

                return weight

            
    def read_dense_complete(self):
        if self.offset == os.path.getsize(self.dense_model_name):
            print("[INFO] all dense weights has been parsed")
        else:
            print("[INFO] not all dense weights has been parsed")