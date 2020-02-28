#coding:utf-8
import numpy as np
import os,pathlib
import torch

""" 项目的根目录 """
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

class Config(object):
    def __init__(self):
        self.point_path = os.path.join(root,"data","百度题库/multi_cls/baidu_95.csv")
        self.stopwords_path = os.path.join(root,"data","stopwords/哈工大停用词表.txt")
        self.w2v_path = os.path.join(root,"data","w2v/w2v_bk.txt")
        self.vocab_path = os.path.join(root,"data","w2v/vocab.pickle")
        self.mlb_path = os.path.join(root,"data","w2v/mlb.pickle")
        self.save_path = os.path.join(root,"data","textcnn_results","multi_cls.h5")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        self.batch_size = 128
        self.embed_dim = 300
        self.max_len = 150
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 128
        self.dense_units = 100
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.max_grad_norm = 2.0
        self.gamma = 0.9
        self.require_improve = 500
        
