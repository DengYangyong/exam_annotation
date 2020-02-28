#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embed_matrix,freeze=False)
        self.convs = nn.ModuleList(
                     [nn.Conv2d(1, config.num_filters, (k, config.embed_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(nn.Linear(config.num_filters * len(config.filter_sizes), config.dense_units),nn.ReLU())
        self.linear = nn.Linear(config.dense_units,config.num_classes)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self,x):
        out = self.embedding(x)
        out = self.dropout(out)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.linear(out)