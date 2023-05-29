#!/usr/bin/env/env python
#!-coding:utf-8 -*-
#!@Time  :2018/7/16 14:37
#!@Author:xiaobai
#!@File  :.py

import os
from datetime import datetime as timefun
import json
import fnmatch
import time  # 导入的time模块和模块中重命名的函数冲突
import csv
from ppg import BASE_DIR
from ppg.utils import exist, load_text, load_json, dump_json, parse_iso_time_string
import xlrd
import sys
import xlwt


def questionnaire_process(sample_rate=64, EDA_sample_rate=4):
    #BASE_DIR = os.path.abspath(r'F:\data\2018_data\formal_experiments')              # 工程路径
    #raw_meta_data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'meta')
    # raw_E4_dir = os.path.join(BASE_DIR, 'data', 'raw_calculate')
    # segmented_data_dir = os.path.join(BASE_DIR, 'data', 'segmented_calculate_rest_5min')

    # E4_BVP_segment = os.path.join(BASE_DIR,  'data','E4_BVP')
    # E4_EDA_segment = os.path.join(BASE_DIR, 'data','E4_EDA')
    # questionnaire_path_dir = os.path.join(BASE_DIR, 'data','questionnaire')
    questionnaire_path_dir = os.path.join(BASE_DIR, 'data', 'questionnaire')

    # if not os.path.exists(E4_BVP_segment):
    #     os.mkdir(E4_BVP_segment)
    # if not os.path.exists(E4_EDA_segment):
    #     os.mkdir(E4_EDA_segment)


    # 将score文本每行取出存入列表
    if exist(pathname=questionnaire_path_dir):
        list = []
        ttt = ['name', 'gender', 'age', 'task1_difficult','task2_difficult', 'task1_stress', 'task2_stress', 'task1_concentrate',
                'task2_concentrate']
        list.append(ttt)
        for filename_with_ext in fnmatch.filter(os.listdir(questionnaire_path_dir), '*.xlsx'):
            ttt=['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
            subject = os.path.splitext(filename_with_ext)[0]
            pathname = os.path.join(questionnaire_path_dir, filename_with_ext)
            print(pathname)
            questionnaire_data = xlrd.open_workbook(pathname,encoding_override="cp1252")
            table = questionnaire_data.sheets()[0]
            ttt[0] = table.cell_value(3, 1) #name
            ttt[1] = table.cell_value(4, 1) #gender
            ttt[2] = table.cell_value(5, 1)  # age
            ttt[3] = table.cell_value(11, 2)  # task1_difficult
            ttt[4] = table.cell_value(11, 3)  # task2_difficult
            ttt[5] = table.cell_value(13, 2)  # task1_stress
            ttt[6] = table.cell_value(13, 3)  # task2_stress
            ttt[7] = table.cell_value(15, 2)  # task1_concentrate
            ttt[8] = table.cell_value(15, 3)  # task2_concentrate
            list.append(ttt)



        filename = xlwt.Workbook()
        sheet = filename.add_sheet('all')
        listlength = len(list)
        rowlenght = len(list[0])
        for rownum in range(0, listlength):
            for num2 in range(0, rowlenght):
                sheet.write(rownum + 3, num2, list[rownum][num2])
        filename.save('questionnaire.xls')







if __name__ == '__main__':
    questionnaire_process()
