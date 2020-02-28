#coding:utf-8
import torch
import numpy as np
from config import Config
import pickle
from data_loader import sentence_proc


def load_pickle(file_path):
    """ 
    用于加载 python的pickle对象 
    """
    return pickle.load(open(file_path,'rb'))

class Vocab:
    def __init__(self,vocab_path):
        """
        加载词表，用于把词转化为ID
        """
        self.word2id = load_pickle(vocab_path)
    
    """ 如果词不在词表中，就转化为 '<unk>' """
    def w2i(self, word):
        if word not in self.word2id:
            return self.word2id["<unk>"]
        return self.word2id[word]


class TextcnnPredict:
    def __init__(self, device="gpu"):
        """
        模型预测，可以选择使用gpu还是cpu
        : param: self.mlb  用于将预测结果转化为多标签
        """
        self.config = Config()
        self.device = device
        self.vocab = Vocab(self.config.vocab_path)
        self.mlb = load_pickle(self.config.mlb_path)
        self.model = self.load_model()
        self.model.eval()
    
    """ 加载为gpu还是cpu版的模型 """
    def load_model(self):
        if self.device == "cpu":
            model = torch.load(self.config.save_path, map_location="cpu")
        else:
            model = torch.load(self.config.save_path)
        return model
    
    """ 把试题进行分词，并转化为id，进行pad """
    def text_to_ids(self,sentence):
        words = sentence_proc(sentence).split()
        words = words[: self.config.max_len]
        ids = [self.vocab.w2i(w) for w in words]
        
        """ 按最大长度进行pad """
        ids += [self.vocab.w2i("<pad>")] * (self.config.max_len - len(words))
        if self.device == "cpu":
            ids = torch.LongTensor([ids])
        else:
            ids = torch.LongTensor([ids]).to(self.config.device)
        return ids    

    def predict(self,text):
        
        with torch.no_grad():
            
            ids = self.text_to_ids(text)
            outputs = self.model(ids)
            
            """ 用sigmoid函数转化为概率分布 """
            outputs = torch.sigmoid(outputs)
            outputs = outputs.data.cpu().numpy()
            
            """ 将概率转化为数值化标签，再转化为多分类的标签 """
            predicts = np.where(outputs > 0.5, 1, 0)             
            labels = self.mlb.inverse_transform(predicts)
            
        return labels[0]


if __name__ == "__main__":

    text = "菠菜从土壤中吸收的氮元素可以用来合成（）A.淀粉和纤维素B.葡萄糖和DNAC.核酸和蛋白质D.麦芽糖和脂肪酸"
    real_label = "高中 生物 分子与细胞 组成细胞的化学元素 组成细胞的化合物"
    
    model = TextcnnPredict(device="cpu")
    predict_label = model.predict(text)
    print("\nPredited label is %s \n" % " ".join(predict_label))
    
    """ 
    Predited label is 分子与细胞 生物 组成细胞的化合物 组成细胞的化学元素 高中 
    """


