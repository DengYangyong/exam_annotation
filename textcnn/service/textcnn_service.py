#coding:utf-8
from sanic import Sanic
from sanic import response
import json
import time
from sanic.exceptions import NotFound
from service_helper import TextcnnServer

""" 定义 ip和端口号 """
app = Sanic(__name__)
ip, port = "0.0.0.0", 6060

""" 路由 (ExmaServer) 错误时，返回错误信息 """
@app.exception(NotFound)
async def url_404(request, excep):
    return response.json({"Error":excep})

""" 定义路由（ExmaServer）和请求方式（POST) """
@app.route('/ExamServer',methods=['POST'])
async def model_server(request):
    try:
        request_json = request.body
        input_json = json.loads(request_json.decode('utf8'))
        result = TextcnnServer(device="gpu").get_result(input_json)
    except Exception as e:
        result = {"code": 400, "message": "预测失败", "Error": e}
    return response.json(result)


if __name__ == '__main__':
    app.run(host=ip,port=port)