#coding:utf-8
import torch
from torchtext import data
import os,re,jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
import matplotlib.pyplot as plt 


""" 一:清洗文本并划分数据集 """
def build_dataset():
    
    """ 读取数据 """
    print("\nLoading the dataset ... \n")
    ancient_his_df = pd.read_csv(os.path.join(config.his_dir_origin,'古代史.csv'))
    contemporary_his_df = pd.read_csv(os.path.join(config.his_dir_origin,'现代史.csv'))
    modern_his_df = pd.read_csv(os.path.join(config.his_dir_origin,'近代史.csv'))
    
    """ 贴标签 """ 
    print("\nLabeling the dataset ... \n")
    ancient_his_df['label'] = '古代史'
    contemporary_his_df['label'] = '现代史'
    modern_his_df['label'] = '近代史'
    
    """ 数据清洗和分词 """ 
    print("\nCleaning text and segmenting ... \n")
    ancient_his_df['item'] = ancient_his_df['item'].apply(sentence_proc)
    contemporary_his_df['item'] = contemporary_his_df['item'].apply(sentence_proc)
    modern_his_df['item'] = modern_his_df['item'].apply(sentence_proc)
    
    """ 划分数据集并保存 """ 
    print("\nMerging and spliting dataset ... \n")
    dataset_df = pd.concat([ancient_his_df,contemporary_his_df,modern_his_df],axis=0,sort=True)
    print(f"\nThe shape of the dataset : {dataset_df.shape}\n")
    
    train_data, test_data = train_test_split(dataset_df[['item','label']],test_size=0.15)
    train_data, valid_data = train_test_split(train_data, test_size=0.15)
    
    train_data.to_csv(os.path.join(config.his_dir_proc,'train.csv'),header=True, index=False)
    valid_data.to_csv(os.path.join(config.his_dir_proc,'valid.csv'),header=True, index=False)
    test_data.to_csv(os.path.join(config.his_dir_proc,'test.csv'),header=True, index=False)

""" 加载哈工大停用词表 """
def load_stop_words(stop_word_path):

    file = open(stop_word_path, 'r', encoding='utf-8')
    stop_words = file.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

""" 进行文本清洗，并用jieba分词 """
def clean_sentence(line):
    line = re.sub(
            "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '',line)
    words = jieba.cut(line, cut_all=False)
    return words

""" 文本清洗，分词和去除停用词 """
def sentence_proc(sentence):
    words = clean_sentence(sentence)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)  

    
""" 二: 构造 n-gram 特征，并生成 batch 迭代器 """   
def batch_generator(config):
    
    """ 定义Field对象 """ 
    print("\nDefining the Field ... \n")
    tokenizer = lambda x: x.split()
    TEXT = data.Field(sequential=True, 
                      tokenize=tokenizer,
                      preprocessing=gene_ngram,
                      include_lengths=False)
    
    LABEL = data.LabelField(sequential=False, dtype=torch.int64)
    fields = [('item', TEXT),('label', LABEL)]
    
    """ 加载CSV数据，建立词汇表 """ 
    print("\nBuilding the vocabulary ... \n")
    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = config.his_dir_proc,
                                            train = 'train.csv',
                                            validation = 'valid.csv',
                                            test = 'test.csv',
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = True) 
    
    """ 千万注意，创建成功后 ，vocab size 为 max features 加 2
     而embedding层的input size 为 vocab size。"""
    TEXT.build_vocab(train_data,
                     max_size=config.max_features)
    LABEL.build_vocab(train_data)
    print(f"\nUnique tokens in TEXT vocabulary: {len(TEXT.vocab)}\n")
    print(f"\nUnique tokens in LABEL vocabulary: {len(LABEL.vocab)}\n") 
    print(LABEL.vocab.stoi)
    print(f"\nInput size or vocab size is {len(TEXT.vocab)}\n")
    print(f"\nPad index is {TEXT.vocab.stoi[TEXT.pad_token]}")
    
    """ 生成batch """ 
    print("\nCreating the batch ... \n")
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
                                          (train_data, valid_data, test_data), 
                                          batch_sizes = (config.batch_size,) * 3,
                                          # 这里注意加一行，而且 item 是之前在Field里面定义好的
                                          sort_key = lambda x: len(x.item),
                                          sort_within_batch=False,
                                          device=config.device)
    
    return train_iter, valid_iter, test_iter

""" 构造n-gram 特征 """
def gene_ngram(sentence,n=3,m=1):
    """
    ----------
    sentence: 分词后的句子
    n: 取3，则为3-gram
    m: 取1，则保留1-gram
    ----------
    """
    if len(sentence) < n:
        n = len(sentence)
    list_ngram=[sentence[i - k:i] for k in range(m, n + 1) for i in range(k, len(sentence) + 1) ]
    ngram = [' '.join(i) for i in list_ngram]
    return ngram



""" 三：计算类别权重，缓解类别不平衡问题 """    
def calcu_class_weights(config):
    
    """ 读取标签数据并转化为数字（非one-hot） """ 
    train_data = pd.read_csv(os.path.join(config.his_dir_proc,"train.csv"))
    labels = train_data["label"].map(config.class_map)
    labels = np.array(labels.tolist(),dtype=np.int32)
    
    """ 计算class weights """ 
    freqs = np.bincount(labels)
    
    """ 作图观察类别不平衡情况 """ 
    visualize_freqs(freqs)
    
    p_class = freqs / len(labels)
    class_weights = 1 / np.log(1.02 + p_class)
    
    class_weights = torch.FloatTensor(class_weights).to(config.device)
    return class_weights
 
""" 观察是否存在类别不平衡问题 """
def visualize_freqs(freq):
    plt.bar(range(3),freq,width=0.25,color=['r','b','y'],label="history")
    plt.xticks(range(3),[0,1,2])
    plt.title("The frequencies of three classes")
    plt.legend()
    plt.show()
    plt.clf()    


if __name__ == "__main__":
    
    config = Config()
    
    """ 划分数据集并保存 """ 
    stop_words = load_stop_words(config.stopwords_path)
    build_dataset()
    
    """ 计算不平衡样本的class weights """ 
    class_weights = calcu_class_weights(config)
    
    """ 生成batch """ 
    train_iter, valid_iter, test_iter = batch_generator(config)
    for x_batch, y_batch in train_iter:
        print(f"The shape of item batch is {x_batch.shape}")
        print(f"The shape of label batch is {y_batch.shape}")
    
    """
    The format of shape is [sequence lengths, batch size], which is different from tensorflow.
    
    The shape of item batch is torch.Size([573, 64])
    The shape of label batch is torch.Size([64])
    """

