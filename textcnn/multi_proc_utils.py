# -*- coding:utf-8 -*-
# Created by LuoJie at 11/17/19

import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores


def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    """
    # 数据切分
    data_split = np.array_split(df, partitions)
    # 线程池
    pool = Pool(cores)
    # 数据分发 合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭线程池
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()
    return data
