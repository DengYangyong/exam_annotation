#coding:utf-8
from sklearn.metrics import f1_score,accuracy_score
from config import Config
import os
from sklearn.preprocessing import MultiLabelBinarizer

config = Config()

all_labels = open(config.class_path,encoding="utf-8").readlines()
all_labels = [label.strip() for label in all_labels]
mlb = MultiLabelBinarizer()
mlb.fit([[label] for label in all_labels])


true_file = os.path.join(config.proc_dir,"test","label.txt")
predict_file = os.path.join(config.output_dir,"epochs3","predicted_label.txt")

y_true, y_pred = [], []
with open(true_file, encoding='utf8') as f:
    for line in f.readlines():
        y_true.append(line.strip().split())

with open(predict_file, encoding='utf8') as f:
    for i,line in enumerate(f.readlines()): 
        y_pred.append(line.strip().split())
        
y_true = mlb.transform(y_true)
y_pred = mlb.transform(y_pred)

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred,average='macro')
f1_micro = f1_score(y_true, y_pred,average='micro')

print("1: Accuracy of model is {:.4f}\n".format(accuracy))
print("2: F1-macro of model is {:.4f}\n".format(f1_macro))
print("3: F1-micro of model is {:.4f}\n".format(f1_micro))
