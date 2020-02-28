#coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time,os
from sklearn.metrics import classification_report, confusion_matrix
from datetime import timedelta

""" 记录训练时间 """
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time 
    return timedelta(seconds=int(round(time_dif)))  

""" 网络的参数进行 xavier 初始化 """            
def init_network(model, method="xavier", exclude="embedding"):
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                nn.init.xavier_normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)

""" 评估函数 """             
def evaluate(config, model, data_iter, test=False):
    
    """ 验证和测试时切换到evaluate模式，结束后再切换为train模式 """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    """ 验证和预测时不需要计算梯度 """
    with torch.no_grad():
        
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            
            """ 把Tensor数据格式转化为numpy格式"""
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    """ 用F1值(宏平均)作为early stop 监控指标 """
    F1_score = metrics.f1_score(labels_all, predict_all,average='macro')
    
    if test:
        
        print("1: 混淆矩阵为：\n")
        print(confusion_matrix(labels_all, predict_all))
        print("\n2: 准确率、召回率和F1值为：\n")
        print(classification_report(labels_all, predict_all, target_names=list(config.class_map.keys()), digits=4))
        print("\n3: F1-score of model is {:.4f}".format(F1_score))
        return F1_score
   
    return F1_score, loss_total / len(data_iter)

           
""" 模型训练函数 """
def train_model(config, model, train_iter, valid_iter, test_iter, class_weights):
    start_time = time.time()
    model.train()
    
    """ 定义优化器，进行梯度裁剪，和学习率衰减 """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    """ 用early stop 防止过拟合 """
    total_batch = 0  
    valid_best_F1 = float('-inf')
    last_improve = 0  
    flag = False 
    save_path = os.path.join(config.save_dir,"his.h5")
    
    for epoch in range(config.num_epochs):
        scheduler.step()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        
        """ 梯度清零，计算loss，反向传播 """
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            loss.backward()
                        
            optimizer.step()
            
            """ 每训练10个batch 就验证一次，如果有提升，就保存并测试一次 """
            if total_batch % 10 == 0:
                
                valid_F1, valid_loss = evaluate(config, model, valid_iter, test=False)
                if valid_F1 > valid_best_F1:
                    evaluate(config, model, test_iter, test=True)
                    valid_best_F1 = valid_F1
                    torch.save(model, save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                    
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Val F1-score: {:.4f} | Time: {} | {}'
                print(msg.format(total_batch, loss.item(), valid_loss, valid_F1, time_dif, improve))
                
                model.train()
                
            total_batch += 1
            if total_batch - last_improve > config.require_improve:
                
                """ 验证集F1超过1000batch没上升，结束训练 """ 
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    
        

