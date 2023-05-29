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


def process(sample_rate=64, EDA_sample_rate=4):
    #BASE_DIR = os.path.abspath(r'F:\data\2018_data\formal_experiments')              # 工程路径
    #raw_meta_data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'meta')
    raw_E4_dir = os.path.join(BASE_DIR, 'data', 'raw_calculate')
    segmented_data_dir = os.path.join(BASE_DIR, 'data', 'segmented_calculate_rest_5min')

    E4_BVP_segment = os.path.join(BASE_DIR,  'data','E4_BVP')
    E4_EDA_segment = os.path.join(BASE_DIR, 'data','E4_EDA')
    scorefile = os.path.join(BASE_DIR, 'data','score_all.txt')

    if not os.path.exists(E4_BVP_segment):
        os.mkdir(E4_BVP_segment)
    if not os.path.exists(E4_EDA_segment):
        os.mkdir(E4_EDA_segment)

    # 将score文本每行取出存入列表
    lines = []
    with open(scorefile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
    print(lines)

    # 每个人对应info字典中包含4个键值：名字、4个section和开始时间
    infos = []
    infos_EDA=[]
    info = {}
    info_EDA={}
    list = []
    ttt = ['name', 'score_1', 'Minued_1', 'ringht_1', 'wrong_1', 'score_2', 'Minued_2', 'ringht_2', 'wrong_2', '', '', '', '', '', '']
    list.append(ttt)

    for i in range(0, len(lines), 10):                  # 每10行都会创建一个info的字典，下一次for循环名字会重复
        rest = lines[i]
        rest_end = lines[i + 1]

        task1 = lines[i + 2]
        task1_end = lines[i + 3]

        nameLine = lines[i + 4]
        scoreline_1 = lines[i+4]

        relax = lines[i + 5]
        relax_end = lines[i + 6]

        task2 = lines[i + 7]
        task2_end = lines[i + 8]
        scoreline_2 = lines[i+9]

        fields = nameLine.split()
        if len(fields) > 1:
            name = fields[0].split(':')
            cur_filename = ''
            if len(name) == 2:
                cur_filename = name[1]

            if len(cur_filename) == 0:
                if not (':' in fields[1]):
                    cur_filename = fields[1]

            if len(cur_filename) == 0:
                continue

        if cur_filename != info.get('filename'):
            info = {}
            infos.append(info)
            info['filename'] = cur_filename
            info['sections'] = []

            # 存储到excel表格，用表格计算
            ttt = [cur_filename, '', '', '', '', '', '', '', '', '', '', '', '', '', '']
            list.append(ttt)

            #解析task_1的performance数据
            scoreline_1s = scoreline_1.split()
            ttt[1] = scoreline_1s[1].split(':')[1]
            ttt[2] = scoreline_1s[2].split(':')[1]
            ttt[3] = scoreline_1s[3].split(':')[1]
            ttt[4] = scoreline_1s[4].split(':')[1]

            # info['score_1'] = int(scoreline_1s[1].split(':')[1])
            # info['Minued_1'] = int(scoreline_1s[2].split(':')[1])
            # info['ringht_1'] = int(scoreline_1s[3].split(':')[1])
            # info['wrong_1'] = int(scoreline_1s[4].split(':')[1])
            #
            # #解析task_2的performance数据
            scoreline_2s = scoreline_2.split()
            ttt[5] = scoreline_2s[1].split(':')[1]
            ttt[6] = scoreline_2s[2].split(':')[1]
            ttt[7] = scoreline_2s[3].split(':')[1]
            ttt[8] = scoreline_2s[4].split(':')[1]

            # info['score_2'] = int(scoreline_2s[1].split(':')[1])
            # info['Minued_2']  = int(scoreline_2s[2].split(':')[1])
            # info['ringht_2 '] = int(scoreline_2s[3].split(':')[1])
            # info['wrong_2 '] = int(scoreline_2s[4].split(':')[1])

            filename = xlwt.Workbook()
            sheet = filename.add_sheet('all')
            listlength = len(list)
            rowlenght = len(list[0])
            for rownum in range(0, listlength):
                for num2 in range(0, rowlenght):
                    sheet.write(rownum + 3, num2, list[rownum][num2])
            filename.save('performance.xls')

            # #输出txt文件
            # outfile = sectionName + '.txt'
            # print(outfile)
            # outfile_path = os.path.join(E4_BVP_segment, outfile)
            # with open(outfile_path, 'w', encoding='utf-8') as f:
            #     for i in sectionData:
            #         f.write(str(i))
            #         f.write('\n')
            # index = index + 1





if __name__ == '__main__':
        process()
