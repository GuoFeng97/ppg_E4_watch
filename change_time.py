#!/usr/bin/env/env python
#!-coding:utf-8 -*-


import os
from datetime import datetime as timefun
import json
import fnmatch
import time  # 导入的time模块和模块中重命名的函数冲突
import csv
from ppg import BASE_DIR
from ppg.utils import exist, load_text, load_json, dump_json, parse_iso_time_string



if __name__ == '__main__':
    # 时间戳转化为带有格式的时间
    pathname = 'BVP.csv'
    if os.path.exists(pathname):  # 路径必须把所有的根目录全部写出来
        with open(pathname) as f:
            data_lines = f.readlines()  # 这里的lines不能上面的重名（可以直接用上面的时间戳，不必要重复写）
            participant_time_TMP = time.localtime(int(float(data_lines[0].strip('\n'))))
            data_starttime = time.strftime("%Y/%m/%d %H:%M:%S", participant_time_TMP)
            E4_data_starttime = timefun.strptime(data_starttime, "%Y/%m/%d %H:%M:%S")
            print(E4_data_starttime)