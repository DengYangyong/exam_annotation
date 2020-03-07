#coding:utf-8
import numpy as np
import os,pathlib


root = pathlib.Path(os.path.abspath(__file__)).parent.parent


class Config(object):
    def __init__(self):
        self.his_origin_dir = os.path.join(root,"data","百度题库/高中_历史/origin")
        self.his_proc_dir = os.path.join(root,"data","bert_multi_cls_results","高中_历史","proc")
        self.output_dir = os.path.join(root,"data","bert_multi_cls_results")
        self.vocab_file = os.path.join("pretrained_model","chinese_L-12_H-768_A-12","vocab.txt")
        
        self.max_len = 150
        self.output_dim = 3
        self.learning_rate = 2e-5
        self.num_epochs = 5
        
