#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config


class FastText(nn.Module):
    def __init__(self,vocab_size, embed_dim, output_dim, pad_idx):
        super(FastText,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, output_dim)
        
    def forward(self, text):
        embed_text = self.embedding(text)
        embed_text = embed_text.permute(1,0,2)
        pooled = F.avg_pool2d(embed_text, (embed_text.shape[1],1)).squeeze(1)
        return self.fc(pooled)
    
if __name__ == "__main__":
    config = Config()
    
    """ 模型初始化 """
    model = FastText(config.vocab_size, 
                     config.embed_dim, 
                     config.output_dim, 
                     config.pad_idx)
    
    """ 设置为训练模式 """
    model.train()
    
    """ 生成输入数据，注意数据类型为torch.long，也就是int64，不然报错 """
    max_lengths = 4
    text = torch.ones(max_lengths,config.batch_size,dtype=torch.long)
    output = model(text)
    print("The output shape is (batch size, class number):{}".format(output.shape))
    
        
   
        
        