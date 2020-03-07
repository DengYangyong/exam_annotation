#coding:utf-8
import sys
sys.path.append("bert")
import pandas as pd
import os
from bert import tokenization
from sklearn.model_selection import train_test_split
from config import Config

config = Config()

""" 一： 读取数据、贴标签和划分数据集 """
def load_dataset():
    
    """ 1: 读取数据 """
    print("\n读取数据 ... \n")
    ancient_his_df = pd.read_csv(os.path.join(config.his_origin_dir,'古代史.csv'))
    contemp_his_df = pd.read_csv(os.path.join(config.his_origin_dir,'现代史.csv'))
    modern_his_df = pd.read_csv(os.path.join(config.his_origin_dir,'近代史.csv'))
    
    """ 2: 贴标签 """ 
    print("\n贴标签 ... \n")
    ancient_his_df['label'] = '古代史'
    contemp_his_df['label'] = '现代史'
    modern_his_df['label'] = '近代史'
    
    """ 3: 划分数据集并保存 """ 
    print("\n划分数据集并保存 ... \n")
    his_df = pd.concat([ancient_his_df,contemp_his_df,modern_his_df],axis=0,sort=True)
    print(f"\nThe shape of the dataset : {his_df.shape}\n")   
    
    df_train, df_test = train_test_split(his_df[:], test_size=0.2, shuffle=True)
    df_valid, df_test = train_test_split(df_test[:], test_size=0.5, shuffle=True)    
    
    return df_train, df_valid, df_test


""" 二：创建保存训练集、验证集和测试集的目录 """
def create_dir():
    
    print("\n创建保存训练集、验证集和测试集的目录 ... \n")
    proc_dir = config.his_proc_dir
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
        print(f'datasize: {len(df_data)}')
        
        """ 打开文件 """
        text_f = open(os.path.join(proc_dir, set_type,"text.txt"), "w",encoding='utf-8')
        token_in_f = open(os.path.join(proc_dir, set_type, "token_in.txt"),"w",encoding='utf-8')
        label_f = open(os.path.join(proc_dir, set_type,"label.txt"), "w",encoding='utf-8')
        
        """ 按字进行切分 """
        text = '\n'.join(df_data.item)
        text_tokened = df_data.item.apply(bert_tokenizer.tokenize)
        text_tokened = '\n'.join([' '.join(row) for row in text_tokened])
        label = '\n'.join(df_data.label)
        
        """ 写入文件 """
        text_f.write(text)
        token_in_f.write(text_tokened)
        label_f.write(label)

        text_f.close()
        token_in_f.close()
        label_f.close()
        
if __name__ == "__main__":
    
    prepare_dataset()
