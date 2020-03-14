#coding:utf-8
import numpy as np
import os,pathlib


root = pathlib.Path(os.path.abspath(__file__)).parent.parent


class Config(object):
    def __init__(self):
        self.point_path = os.path.join(root,"data","百度题库/multi_cls/baidu_95.csv")
        self.proc_dir = os.path.join(root,"data","bert_multi_label_results","proc")
        self.class_path = os.path.join(root,"data","百度题库/multi_cls","95_class.txt")
        self.output_dir = os.path.join(root,"data","bert_multi_label_results")
        self.vocab_file = os.path.join("pretrained_model","roberta_zh_l12","vocab.txt")
        
        self.max_len = 150
        self.output_dim = 95
        self.learning_rate = 2e-5
        self.num_epochs = 3
