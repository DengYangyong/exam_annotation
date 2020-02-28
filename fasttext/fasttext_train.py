#coding:utf-8
import torch
import torch.optim as optim
from config import Config
from fasttext_model import FastText
from data_loader import calcu_class_weights,batch_generator
from fasttext_train_helper import init_network, train_model
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

""" 统计参数量 """
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

""" 设置随机数种子 """
def set_manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" 模型训练 """
def train(config):
    
    set_manual_seed(10)
    
    INPUT_DIM = config.vocab_size
    EMBED_DIM = config.embed_dim
    OUTPUT_DIM = config.output_dim
    PAD_IDX = config.pad_idx
    
    print("Building the fasttext model ... \n")
    model = FastText(INPUT_DIM,EMBED_DIM,OUTPUT_DIM,PAD_IDX)
    print(f'The model has {count_params(model):,} trainable parameters\n')
    
    model.to(config.device)
    
    print("Calculate class weigths ... \n")
    class_weights = calcu_class_weights(config)
    
    print("Preparing the batch data ... \n")
    train_iter, valid_iter, test_iter = batch_generator(config)
    
    init_network(model)
    train_model(config, model, train_iter, valid_iter, test_iter, class_weights)

if __name__ == "__main__":
    config = Config()
    train(config)