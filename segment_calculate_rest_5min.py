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
    for i in range(0, len(lines), 10):                  # 每10行都会创建一个info的字典，下一次for循环名字会重复
        rest = lines[i]
        rest_end = lines[i + 1]

        task1 = lines[i + 2]
        task1_end = lines[i + 3]

        nameLine = lines[i + 4]

        relax = lines[i + 5]
        relax_end = lines[i + 6]

        task2 = lines[i + 7]
        task2_end = lines[i + 8]

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

            info_EDA={}
            infos_EDA.append(info_EDA)
            info_EDA['filename'] = cur_filename
            info['sections'] = []
            info_EDA['sections']=[]

        if os.path.exists(raw_E4_dir):
            for E4_file in os.listdir(raw_E4_dir):                                      #轮询每个人的数据文件夹
                participant_name, file_ext = os.path.splitext(E4_file)
                lis = participant_name.split('_')
                particpant = lis[0]

                if particpant != str(info['filename']):
                    continue

                time_TMP = lis[1]                                                    #每个文件夹中的时间戳
                info['E4_data_starttime'] = time_TMP
                info_EDA['E4_data_starttime'] = time_TMP

                # 时间戳转化为带有格式的时间
                filename = 'BVP.csv'
                E4_file_dir = str(info['filename']) + '_' + time_TMP + '_' + 'A01671'
                E4_file_data_dir = os.path.join(E4_file_dir, filename)
                pathname = os.path.join(raw_E4_dir, E4_file_data_dir)
                if os.path.exists(pathname):                                                                                    # 路径必须把所有的根目录全部写出来
                    with open(pathname) as f:
                        data_lines = f.readlines()                                                                                  #这里的lines不能上面的重名（可以直接用上面的时间戳，不必要重复写）
                        participant_time_TMP = time.localtime(int(float(data_lines[0].strip('\n'))))
                        data_starttime = time.strftime("%Y/%m/%d %H:%M:%S", participant_time_TMP)
                        E4_data_starttime = timefun.strptime(data_starttime, "%Y/%m/%d %H:%M:%S")
                        print(E4_data_starttime)

                        # rest索引
                        fields = rest.split()
                        fields_end = rest_end.split()
                        if len(fields) > 1:
                            cur_rest = timefun.strptime(fields[0].split(':')[1] + ' ' + fields[1],"%Y/%m/%d %H:%M:%S")  # rest开始是对应的时间
                            rest_end_time = timefun.strptime(fields_end[0].split(':')[1] + ' ' + fields_end[1], "%Y/%m/%d %H:%M:%S")
                            if info.get('startTime') == None or info.get('startTime') > cur_rest:
                                info['startTime'] = cur_rest
                                info_EDA['startTime'] = cur_rest
                                print(cur_rest)
                            rest_Index = ((cur_rest - E4_data_starttime).seconds +300)* sample_rate
                            # rest_end_index = (rest_end_time - E4_data_starttime).seconds * sample_rate
                            rest_end_index = rest_Index + 300 * sample_rate  #把休息时间设置为中间的5分钟,开始以后5分钟后计算

                            EDA_rest_Index = (cur_rest - E4_data_starttime).seconds * EDA_sample_rate
                            EDA_rest_end_index = (rest_end_time - E4_data_starttime).seconds * EDA_sample_rate

                        # task1索引
                        fields = task1.split()
                        fields_end = task1_end.split()
                        if len(fields) > 1:
                            cur_task1 = timefun.strptime(fields[0].split(':')[1] + ' ' + fields[1], "%Y/%m/%d %H:%M:%S")
                            task1_end_time = timefun.strptime(fields_end[0].split(':')[1] + ' ' + fields_end[1], "%Y/%m/%d %H:%M:%S")
                            task1_Index = (cur_task1 - E4_data_starttime).seconds * sample_rate
                            task1_end_index = (task1_end_time - E4_data_starttime).seconds * sample_rate

                            EDA_task1_Index = (cur_task1 - E4_data_starttime).seconds * EDA_sample_rate
                            EDA_task1_end_index = (task1_end_time - E4_data_starttime).seconds * EDA_sample_rate

                        # relax索引
                        fields = relax.split()
                        fields_end = relax_end.split()
                        if len(fields) > 1:
                            cur_relax = timefun.strptime(fields[0].split(':')[1] + ' ' + fields[1], "%Y/%m/%d %H:%M:%S")
                            relax_end_time = timefun.strptime(fields_end[0].split(':')[1] + ' ' + fields_end[1], "%Y/%m/%d %H:%M:%S")
                            relax_Index = (cur_relax - E4_data_starttime).seconds * sample_rate
                            relax_end_index = (relax_end_time - E4_data_starttime).seconds * sample_rate

                            EDA_relax_Index = (cur_relax - E4_data_starttime).seconds * EDA_sample_rate
                            EDA_relax_end_index = (relax_end_time - E4_data_starttime).seconds * EDA_sample_rate

                        # task2索引
                        fields = task2.split()
                        fields_end = task2_end.split()
                        if len(fields) > 1:
                            cur_task2 = timefun.strptime(fields[0].split(':')[1] + ' ' + fields[1], "%Y/%m/%d %H:%M:%S")
                            task2_end_time = timefun.strptime(fields_end[0].split(':')[1] + ' ' + fields_end[1],"%Y/%m/%d %H:%M:%S")
                            task2_Index = (cur_task2 - E4_data_starttime).seconds * sample_rate
                            task2_end_index = (task2_end_time - E4_data_starttime).seconds * sample_rate

                            EDA_task2_Index = (cur_task2 - E4_data_starttime).seconds * EDA_sample_rate
                            EDA_task2_end_index = (task2_end_time - E4_data_starttime).seconds * EDA_sample_rate

                        # 由于手环时间没有错误，直接分成4段，不需要判断时间落点
                        info['sections'].append(slice(rest_Index, rest_end_index))
                        info['sections'].append(slice(task1_Index, task1_end_index))
                        info['sections'].append(slice(relax_Index, relax_end_index))
                        info['sections'].append(slice(task2_Index, task2_end_index))

                        info_EDA['sections'].append(slice(EDA_rest_Index, EDA_rest_end_index))
                        info_EDA['sections'].append(slice(EDA_task1_Index, EDA_task1_end_index))
                        info_EDA['sections'].append(slice(EDA_relax_Index, EDA_relax_end_index))
                        info_EDA['sections'].append(slice(EDA_task2_Index, EDA_task2_end_index))
                        print(info)
                        print(info_EDA)
                            # break

    # 轮询info，读取BVP文本，产生切割后txt,和 json文件
    output_data = {}
    for info in infos:
        output = {}
        output['filename'] = str(info['filename'])
        participant = str(info['filename']) #志愿者姓名
        if participant not in output_data:
            output_data[participant] = {}

        filename = 'BVP.csv'
        print(info)
        E4_file_dir = str(info['filename']) + '_' + str(info['E4_data_starttime']) + '_' + 'A01671'
        pathname = os.path.join(E4_file_dir, filename)
        pathname=os.path.join(raw_E4_dir,pathname)              #取路径时间戳对不上
        data = []
        if os.path.exists(pathname):
            with open(pathname) as f:
                lines=f.readlines()
                for line in lines[1:]:
                    data.append(int(float(line.strip())))          #BVP的每一行数据存入数组中

        index = 0
        for section in info.get('sections'):  # for循环取出infos中的sections
            sectionData = data[section]  #BVP数据
            sectionName = str(info['filename']) + '_E4_BVP_session' + '%s' % index  # 切割BVP后对应的4个txt对应的文件名
            session_id = str(index)
            #产生json数据
            #output_data[participant][session_id]['ppg']['sample_rate'] = PPG_SAMPLE_RATE
            output_data[participant][session_id] = {
                'ppg': {
                    'sample_rate': None,
                    'signal': None,
                }
            }
            output_data[participant][session_id]['ppg']['signal'] = sectionData
            output_data[participant][session_id]['ppg']['sample_rate'] = 64

            # #输出txt文件
            # outfile = sectionName + '.txt'
            # print(outfile)
            # outfile_path = os.path.join(E4_BVP_segment, outfile)
            # with open(outfile_path, 'w', encoding='utf-8') as f:
            #     for i in sectionData:
            #         f.write(str(i))
            #         f.write('\n')
            index = index + 1

        #输出json文件，每次for循环都向json中存入对应的数据块
        # Save segmented signal data
        for participant in output_data:
            output_filename = '%s.json' % participant
            dump_json(data=output_data[participant],
                      pathname=os.path.join(segmented_data_dir, output_filename), overwrite=True)

    # #infos_EDA，读取EDA文本
    # for info1 in infos_EDA:
    #     output_EDA = {}
    #     output_EDA['filename'] = str(info1['filename'])
    #     filename_EDA = 'EDA.csv'
    #     E4_file_dir = str(info1['filename']) + '_' + str(info1['E4_data_starttime']) + '_' + 'A01671'
    #     pathname = os.path.join(E4_file_dir, filename_EDA)
    #     pathname=os.path.join(raw_E4_dir,pathname)
    #     data_EDA = []
    #     if os.path.exists(pathname):
    #         with open(pathname) as f:
    #             lines_EDA = f.readlines()
    #             for line in lines_EDA[1:]:
    #                 data_EDA.append(float(line.strip()))# EDA的每一行数据存入数组中
    #
    #     index_EDA= 0
    #     for section1 in info_EDA.get('sections'):            #这里的info是从infos_EDA取出的
    #         sectionData_EDA = data_EDA[section1]         # EDA数据                #确定的范围超过了文本中的EDA数据，所以导致文生成的文本为空
    #         sectionName_EDA = str(info1['filename']) + '_E4_EDA_session' + '%s' % index_EDA
    #         outfile = sectionName_EDA + '.txt'
    #         print(outfile)
    #         outfile_path = os.path.join(E4_EDA_segment, outfile)
    #         with open(outfile_path, 'w', encoding='utf-8') as f:
    #             for i in sectionData_EDA:
    #                 f.write(str(i))
    #                 f.write('\n')
    #         index_EDA = index_EDA + 1

if __name__ == '__main__':
        process()
