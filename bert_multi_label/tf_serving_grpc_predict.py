# encoding: utf-8
import sys
sys.path.append("bert")
import tokenization
from extract_features import InputExample, convert_examples_to_features
import numpy as np
import requests,os,json,time
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

from sklearn.preprocessing import MultiLabelBinarizer
from config import Config
config = Config()

""" 1: 准备多标签处理工具，用于将概率转为文本标签 """
all_labels = open(config.class_path,encoding="utf-8").readlines()
all_labels = [label.strip() for label in all_labels]

mlb = MultiLabelBinarizer()
mlb.fit([[label] for label in all_labels])

"""2: 初始化tokenizer，用于文本到id的转换"""
vocab_file = os.environ.get('vocab_file', './pretrained_model/roberta_zh_l12/vocab.txt')
max_token_len = os.environ.get('max_token_len', 400)
tokenizer = tokenization.FullTokenizer( vocab_file=vocab_file, do_lower_case=True)

tf.app.flags.DEFINE_string('server', '0.0.0.0:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


"""3: 将样本转化为符合的输入格式"""		
def preprocess(text):

    text_a = text
    example = InputExample(unique_id=None, text_a=text_a, text_b=None)
    feature = convert_examples_to_features([example], max_token_len, tokenizer)[0]
		
    """4: 从上一步的信息可知，输入的key是input_ids，维度是（1,400），同时外包[]表示batch size 为1 """
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return input_ids


"""5: 输入文本，预测文本标签 """
def predict(text):

	input_ids = preprocess(text)
    
	channel = grpc.insecure_channel(FLAGS.server)
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'exam_classify'                             # 对应上图第一个方框
	request.model_spec.signature_name = 'serving_default'                 # 对应上图第二个方框
	request.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids)) # 对应上图第三个方框,并转化为protobuf格式
    
	start = time.time()
	response = stub.Predict.future(request, 10.0)                         # 10秒timeout
	result = response.result()
	probs = tf.make_ndarray(result.outputs["probabilities"])              # 对应上图第四个方框，同时将tensor转化为numpy    
	end = time.time()
	print("\nTime usage is {:.6f}".format(end - start))
    
	"""6: 将概率转化为文本标签"""
	label_ids = np.where(probs > 0.5,1,0)
	label = mlb.inverse_transform(label_ids)[0]
	return label


if __name__ == "__main__":

    text = "菠菜从土壤中吸收的氮元素可以用来合成（）A.淀粉和纤维素B.葡萄糖和DNAC.核酸和蛋白质D.麦芽糖和脂肪酸"
    real_label = "高中 生物 分子与细胞 组成细胞的化学元素 组成细胞的化合物"
    
    predict_label = predict(text)
    print("\nReal label is: %s " % real_label)
    print("\nPredited label is: %s \n" % " ".join(predict_label))
