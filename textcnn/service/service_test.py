# coding=utf8
import os,sys
import json
import socket
import time
import urllib.request
from datetime import timedelta


""" 记录花费的时间 """
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time 
    return time_dif          

""" 测试服务的响应时间 """
def test_service(content, port):
    
    url = 'http://0.0.0.0:{}/ExamServer'.format(port)
    app_data = {"request_id": "ExamServer", "query": content}
    
    """ 转化为json格式 """
    app_data=json.dumps(app_data).encode("utf-8")
    
    start_time = time.time()
    req = urllib.request.Request(url, app_data)
    try:
        """ 调用服务，得到结果 """
        response = urllib.request.urlopen(req)
        response = response.read().decode("utf-8")
        
        """ 从json格式中解析出来 """
        response = json.loads(response)
    except Exception as e:
        print(e)
        response = None
        
    """ 打印耗时 """
    time_usage = get_time_dif(start_time)
    print("Time usage: {}".format(time_usage))
    print(response)
    return time_usage


if __name__=='__main__':
    
    """ 测试1000次，得到平均响应时间 """
    time_usage = 0
    for i in range(1000):
        content = "菠菜从土壤中吸收的氮元素可以用来合成（）A.淀粉和纤维素B.葡萄糖和DNAC.核酸和蛋白质D.麦芽糖和脂肪酸"
        time_usage += test_service(content, 6060)
    print("Time usage average is {}".format( time_usage / 1000))



