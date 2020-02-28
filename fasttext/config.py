#coding:utf-8
import numpy as np
import os,pathlib
import torch

root = pathlib.Path(os.path.abspath(__file__)).parent.parent


class Config(object):
    def __init__(self):
        self.his_dir_origin = os.path.join(root,"data","百度题库/高中_历史/origin")
        self.his_dir_proc = os.path.join(root,"data","百度题库/高中_历史/proc")
        self.class_map = {"现代史":0, "近代史":1, "古代史":2}
        self.stopwords_path = os.path.join(root,"data","stopwords/哈工大停用词表.txt")
        self.save_dir = os.path.join(root,"data","fasttext_results")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.max_features = 30000
        self.vocab_size = 30002
        self.batch_size = 32
        self.pad_idx = 1
        self.embed_dim = 300
        self.output_dim = 3
        self.dropout = 0.5
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.max_grad_norm = 2.0
        self.gamma = 0.9
        self.require_improve = 1000
        
