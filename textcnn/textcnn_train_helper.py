#coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import time,os
from sklearn.metrics import f1_score,accuracy_score
from datetime import timedelta

""" 记录训练时间 """
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time 
    return timedelta(seconds=int(round(time_dif)))  

""" 网络的参数进行xavier初始化，embedding 层不需要 """            
def init_network(model, method="xavier", exclude="embedding"):
    for name, w in model.named_parameters():
        if exclude in name:
            continue
        if "weight" in name:
            nn.init.xavier_normal_(w)
        elif "bias" in name:
            nn.init.constant_(w, 0)


""" 评估函数 """
def evaluate(config, model, data_iter, test=False):
    
    """ 验证时切换到 evaluate 模式，验证结束再切换为 train 模式 """
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    loss_total = 0
    labels_all = []
    predicts_all = []
    
    """ 验证和预测时都不需要计算梯度 """
    with torch.no_grad():
        
        for texts, labels in data_iter:
            outputs = model(texts)
            
            """ 测试和验证时，计算损失不用带权重 """
            loss = criterion(outputs, labels)
            loss_total += loss
            
            """ 把Tensor数格式转化为numpy格式 """
            labels = labels.data.cpu().numpy()
            outputs = torch.sigmoid(outputs)
            outputs = outputs.data.cpu().numpy()
            
            """ 转化为多分类的标签 """
            predicts = np.where(outputs > 0.5, 1, 0)
            
            labels_all += labels.tolist()
            predicts_all += predicts.tolist()
            
    labels_all = np.array(labels_all,dtype=int) 
    predicts_all = np.array(predicts_all,dtype=int)
    
    """ 计算f1值(宏平均和微平均) """
    f1_score_macro = f1_score(labels_all, predicts_all,average='macro')
    f1_score_micro = f1_score(labels_all, predicts_all,average='micro')
    accuracy = accuracy_score(labels_all, predicts_all)
    
    if test:
        print("1: Accuracy of model is {:.4f}\n".format(accuracy))
        print("2: F1-macro of model is {:.4f}\n".format(f1_score_macro))
        print("3: F1-micro of model is {:.4f}\n".format(f1_score_micro))
   
    return f1_score_macro, f1_score_micro, loss_total / len(data_iter)
           

def train_model(config, model, train_iter, valid_iter, test_iter):
    start_time = time.time()
    model.train()
    
    """ 定义损失函数，并传入类别权重 """
    # criterion = nn.BCEWithLogitsLoss(pos_weight=config.class_weights)
    criterion = nn.BCEWithLogitsLoss()
    
    """ 定义优化器，进行梯度裁剪，和学习率衰减 """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    """ 用early stop 防止过拟合 """
    total_batch = 0  
    valid_best_f1 = float('-inf')
    last_improve = 0  
    flag = False 
    
    for epoch in range(config.num_epochs):
        scheduler.step()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        
        """ 梯度清零，计算loss，反向传播 """
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
                                                        
            """ 训练时，计算损失带权重 """
            loss = criterion(outputs, labels)
            loss.backward()
                        
            optimizer.step()
            
            """ 每训练10个batch就验证一次，如果有提升，就保存并测试一次 """
            if total_batch % 10 == 0:
                
                valid_f1_macro, valid_f1_micro, valid_loss = evaluate(config, model, valid_iter, test=False)
                
                """ 以f1微平均作为early stop的监控指标 """
                if valid_f1_macro > valid_best_f1:
                    evaluate(config, model, test_iter, test=True)
                    valid_best_f1 = valid_f1_macro
                    torch.save(model, cofing.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                    
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Val F1-macro: {:.4f} | Val F1-micro: {:.4f} | Time: {} | {}'
                print(msg.format(total_batch, loss.item(), valid_loss, valid_f1_macro, valid_f1_micro, time_dif, improve))
                
                model.train()
                
            total_batch += 1
            if total_batch - last_improve > config.require_improve:
                """ 验证集loss超过500batch没下降，结束训练 """
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    
        

