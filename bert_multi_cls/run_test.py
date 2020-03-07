#coding:utf-8
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from config import Config
import os

config = Config()

all_labels = ["古代史","近代史","现代史"]
labels_map = {label:i for i,label in enumerate(all_labels)}


true_file = os.path.join(config.his_proc_dir,"test","label.txt")
predict_file = os.path.join(config.output_dir,"epochs5","predicted_label.txt")

y_true, y_pred = [], []
with open(true_file, encoding='utf8') as f:
    for line in f.readlines():
        y_true.append(labels_map[line.strip()])

with open(predict_file, encoding='utf8') as f:
    for i,line in enumerate(f.readlines()): 
        y_pred.append(labels_map[line.strip()])
         
f1_macro = f1_score(y_true, y_pred,average='macro')
f1_micro = f1_score(y_true, y_pred,average='micro')

print("1: 混淆矩阵为：\n")
print(confusion_matrix(y_true, y_pred))
print("\n2: 准确率、召回率和F1值为：\n")
print(classification_report(y_true, y_pred, target_names=all_labels, digits=4))
print("\n3: f1-macro of model is {:.4f}".format(f1_macro))
print("\n4: f1-micro of model is {:.4f}".format(f1_micro))

