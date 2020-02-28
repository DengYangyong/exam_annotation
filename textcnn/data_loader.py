#coding:utf-8
import torch
import torch.utils.data as data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os,re,jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from multi_proc_utils import parallelize
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from tqdm import tqdm
import pickle

config = Config() 

""" 一：对样本和标签进行数值化，建立词表 """
def build_dataset(config):
    
    """ 1：加载数据 """
    print("\nLoading the dataset ... \n")
    point_df = pd.read_csv(config.point_path,header=None)
    point_df.dropna(inplace=True)
    point_df = point_df.rename(columns={0:"label",1:"content"})
    print(f"\nThe shape of the dataset : {point_df.shape}\n")
    
    """ 2：开多进程进行数据清洗和分词 """
    print("\nCleaning text and segmenting ... \n")
    point_df = parallelize(point_df,proc)
    
    """ 3：对样本进行 zero pad，并转化为id """
    print("\nZero padding and transfering id ...\n")
    text_tokenizer = Tokenizer(oov_token="<unk>")
    text_tokenizer.fit_on_texts(point_df["content"])
    corpus_x = text_tokenizer.texts_to_sequences(point_df["content"])
    config.max_len = calcu_max_len(point_df)
    corpus_x = pad_sequences(corpus_x,maxlen=config.max_len,padding="post",truncating="post")
    
    """ 4: 对多标签分类的标签进行数值化 """ 
    print("\nNumeralizing the multiclass labels ... \n")
    point_df["label"] = point_df["label"].apply(lambda x:x.split())
    mlb = MLB()
    corpus_y = mlb.fit_transform(point_df["label"])
    config.num_classes = corpus_y.shape[1]
    
    """ 5: 用样本构建词表，加入 <pad> 的id """ 
    print("\nBuilding the vocab ...\n")
    word_index = text_tokenizer.word_index
    vocab = dict({"<pad>":0}, **word_index)
    config.vocab_size = len(vocab)
    
    """ 保存好标签转化器，在模型预测时用 """
    save_pickle(mlb, config.mlb_path)    
    
    """ 保存好词表，模型预测时用 """
    save_pickle(vocab,config.vocab_path)
    
    return corpus_x,corpus_y,vocab

""" 加载停用词 """ 
def load_stop_words(stop_word_path):
    file = open(stop_word_path, 'r', encoding='utf-8')
    stop_words = file.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

stop_words = load_stop_words(config.stopwords_path) 

""" 清洗文本 """ 
def clean_sentence(line):
    line = re.sub(
            "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '',line)
    words = jieba.lcut(line, cut_all=False)
    return words

""" 进行分词 """ 
def sentence_proc(sentence):
    words = clean_sentence(sentence)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def proc(df):
    df["content"] = df["content"].apply(sentence_proc)
    return df

""" 计算输入长度，使其能涵盖95%样本 """ 
def calcu_max_len(df):
    
    df["lengths"] = df["content"].apply(lambda x:x.count(' ')+1)
    max_lengths = max(df["lengths"])
    for len_ in range(50,max_lengths,50):
        bool_ = df["lengths"] < len_
        cover_rate = sum(bool_.apply(int)) / len(bool_)
        if cover_rate >= 0.95:
            return len_


""" 四：计算类别权重，缓解类别不平衡问题 """
def calcu_class_weights(labels,config):
    labels = torch.FloatTensor(labels)
    
    freqs = torch.zeros_like(labels[0])
    for y in labels:
        freqs += y
    
    weights = freqs / len(labels)
    weights = 1 / torch.log(1.01 + weights)
    
    weights = weights.to(config.device)
    return weights   

    
""" 二: 加载预训练词向量，并与词表相对应 """ 
def load_embed_matrix(vocab,config):
    
    """ 1: 加载百度百科词向量 """ 
    print("\nLoading baidu baike word2vec ...\n")
    embed_index = load_w2v(config.w2v_path)
    
    """ 2: 词向量矩阵与词表相对应 """ 
    vocab_size = len(vocab)
    embed_matrix = np.zeros((vocab_size,config.embed_dim))
    for word,index in vocab.items():
        vector = embed_index.get(word)
        if vector is not None:
            embed_matrix[index] = vector
            
    embed_matrix = torch.FloatTensor(embed_matrix)
            
    return embed_matrix
        
""" 加载百度百科词向量 """
def load_w2v(path):
    
    file = open(path,encoding="utf-8")
    
    embed_index = {}
    for i,line in tqdm(enumerate(file)):
        if i == 0:
            continue
        value = line.split()
        word = value[0]
        emb = np.asarray(value[1:], dtype="float32")
        embed_index[word] = emb
        
    return embed_index


""" 三: 划分数据集，并生成 batch 迭代器 """ 
def batch_generator(x,y,size,config):
    
    """ 1: 划分数据集 """ 
    print("\nSpliting the dataset ... \n")
    train_data, valid_data, test_data = split_dataset(x, y, size, config.device)
    
    """ 2: 生成 batch 迭代器 """
    train_iter = batch_iterator(train_data, config.batch_size)
    valid_iter = batch_iterator(valid_data, config.batch_size)
    test_iter = batch_iterator(test_data, config.batch_size)
    
    return train_iter,valid_iter,test_iter  

""" 划分数据集 """ 
def split_dataset(x,y,size,device):
    
    train_x, valid_x, train_y, valid_y = train_test_split(x,y,test_size=size,random_state=10)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=size,random_state=10) 
    
    train_x = torch.LongTensor(train_x).to(device)
    valid_x = torch.LongTensor(valid_x).to(device)
    test_x = torch.LongTensor(test_x).to(device)
    
    train_y = torch.FloatTensor(train_y).to(device)
    valid_y = torch.FloatTensor(valid_y).to(device)
    test_y = torch.FloatTensor(test_y).to(device)    
    
    return (train_x,train_y) , (valid_x,valid_y), (test_x, test_y)
    
""" 重写 dataset 类"""
class MyDataset(data.Dataset):
    def __init__(self, dataset):
        self.X, self.Y = dataset

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

""" 生成 迭代器 """     
def batch_iterator(dataset,batch_size):
    dataset = MyDataset(dataset)
    batcher = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return batcher
 
""" 保存为pickle对象 """ 
def save_pickle(s,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(s,f,protocol=2)
        
""" 加载pickle对象 """ 
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        s = pickle.load(f)
    return s

     
if __name__ == "__main__":
    
    corpus_x, corpus_y, vocab = build_dataset(config)
    
    class_weights = calcu_class_weights(corpus_y,config)
    
    embed_matrix = load_embed_matrix(vocab,config)
    
    train_iter, valid_iter, test_iter = batch_generator(corpus_x, corpus_y, 0.15, config)
    for x_batch, y_batch in train_iter:
        print(f"The shape of content batch is {x_batch.shape}")
        print(f"The shape of label batch is {y_batch.shape}")
        break
