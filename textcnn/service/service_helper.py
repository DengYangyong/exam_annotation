# coding:utf-8
import sys
sys.path.append("../")
from textcnn_predict import TextcnnPredict

""" 服务被调用时的情况 """
message_dic={'200':'正常',
             '300':'请求格式错误',
             '400':'模型预测失败'}

class TextcnnServer:
    def __init__(self,device="gpu"):
        """ 
        把模型的预测函数初始化,
        设置使用CPU还是GPU启动服务.
        """
        self.predict = TextcnnPredict(device).predict
    
    """ 把字典格式的请求数据，解析出来 """
    def parse(self, app_data):
        request_id = app_data["request_id"]
        text = app_data["query"]
        return request_id, text
    
    """ 得到服务的调用结果，包括模型结果和服务的情况 """
    def get_result(self,data):
        code = '200'
        try:
            request_id, text = self.parse(data) 
        except Exception as e:
            print('error info : {}'.format(e))
            code='300'
            request_id = "None"
        try:
            if code == '200':
                label = self.predict(text)
            elif code == '300':
                label = '高中'
        except Exception as e:
            print('error info : {}'.format(e))
            label = '高中'
            code='400'
    
        result = {'label': label,'code':code,'message':message_dic[code],'request_id':request_id}  
        return result

if __name__ == "__main__":
    
    server = TextcnnServer(device="gpu")
    data = {"request_id": "ExamServer", 
            "query" :"菠菜从土壤中吸收的氮元素可以用来合成（）A.淀粉和纤维素B.葡萄糖和DNAC.核酸和蛋白质D.麦芽糖和脂肪酸"}
    print("\n The result is {}".format(server.get_result(data)))
