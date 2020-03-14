#coding:utf-8
import sys
sys.path.append("bert")
import pandas as pd
import os
from bert import tokenization
from sklearn.model_selection import train_test_split
from config import Config
from sklearn.preprocessing import MultiLabelBinarizer as MLB

config = Config()

""" 一： 读取数据、贴标签和划分数据集 """
def load_dataset():
    
    """ 1: 读取数据 """
    print("\n读取数据 ... \n")
    point_df = pd.read_csv(config.point_path, header=None, names=["label", "item"])
    point_df.dropna(inplace=True)
    print(f"\nThe shape of the dataset : {point_df.shape}\n")
    
    """ 2: 获取所有类别并保存 """
    print("\n获取所有类别 ... \n")
    point_df["label"] = point_df["label"].apply(lambda x:x.split())
    mlb = MLB()
    mlb.fit(point_df["label"])
    all_class = mlb.classes_.tolist()
    
    with open(config.class_path, "w",encoding='utf-8') as f:
        f.write("\n".join(all_class))
    
    """ 3: 划分数据集 """ 
    print("\n划分数据集 ... \n")
    df_train, df_test = train_test_split(point_df[:], test_size=0.2, shuffle=True)
    df_valid, df_test = train_test_split(df_test[:], test_size=0.5, shuffle=True)    
    
    return df_train, df_valid, df_test


""" 二：创建保存训练集、验证集和测试集的目录 """
def create_dir():
    
    print("\n创建保存训练集、验证集和测试集的目录 ... \n")
    proc_dir = config.proc_dir
    if not os.path.exists(proc_dir):
        os.makedirs(os.path.join(proc_dir, "train"))
        os.makedirs(os.path.join(proc_dir, "valid"))
        os.makedirs(os.path.join(proc_dir, "test")) 
        
    return proc_dir


""" 三：按字粒度切分样本 """
def prepare_dataset():
    
    """ 1: 读取数据、贴标签和划分数据集"""
    df_train, df_valid, df_test = load_dataset()
    
    """ 2: 创建保存训练集、验证集和测试集的目录"""
    proc_dir = create_dir()
    
    """ 3: 初始化 bert_token 工具"""
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=True) 
    
    """ 4: 按字进行切分"""
    print("\n按字进行切分 ... \n")
    
    type_list = ["train", "valid", "test"]
    for set_type, df_data in zip(type_list, [df_train, df_valid, df_test]):
        print(f'\ndatasize: {len(df_data)}\n')
        
        """ 打开文件 """
        text_f = open(os.path.join(proc_dir, set_type,"text.txt"), "w",encoding='utf-8')
        token_in_f = open(os.path.join(proc_dir, set_type, "token_in.txt"),"w",encoding='utf-8')
        label_f = open(os.path.join(proc_dir, set_type,"label.txt"), "w",encoding='utf-8')
        
        """ 按字进行切分 """
        text = '\n'.join(df_data.item)
        text_tokened = df_data.item.apply(bert_tokenizer.tokenize)
        text_tokened = '\n'.join([' '.join(row) for row in text_tokened])
        label = '\n'.join([" ".join(row) for row in df_data.label])
        
        """ 写入文件 """
        text_f.write(text)
        token_in_f.write(text_tokened)
        label_f.write(label)

        text_f.close()
        token_in_f.close()
        label_f.close()
        
if __name__ == "__main__":
    
    prepare_dataset()
