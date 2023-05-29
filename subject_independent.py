# -*- coding: utf-8 -*-

import os
import fnmatch
from functools import reduce
from ppg import BASE_DIR
from ppg.params import TRAINING_DATA_RATIO
from ppg.utils import exist, load_json, dump_json, get_change_ratio


def merge(feature_data_1, feature_data_2):
    return {
        '0': feature_data_1['0'] + feature_data_2['0'],
        '1': feature_data_1['1'] + feature_data_2['1'],
        '2': feature_data_1['2'] + feature_data_2['2'],
        '3': feature_data_1['3'] + feature_data_2['3'],
    }


def subject_independent():
    preprocessed_data_dir = os.path.join(BASE_DIR, 'data', 'preprocessed_calculate_rest_5min')
    subject_independent_data_dir = os.path.join(BASE_DIR, 'data', 'subject_independent_calculate_rest_5min')

    #首先把所有的数据都存到all_subject_data里面，然后从里面分出训练集和测试集
    if exist(pathname=preprocessed_data_dir):
        all_subject_data = {}
        for filename_with_ext in fnmatch.filter(os.listdir(preprocessed_data_dir), '*.json'):
            subject = os.path.splitext(filename_with_ext)[0]
            feature_data = {
                '0': [],
                '1': [],
                '2': [],
                '3': [],
            }
            pathname = os.path.join(preprocessed_data_dir, filename_with_ext)
            json_data = load_json(pathname=pathname)
            if json_data is not None:
                for session in json_data:
                        feature_data[session].append({
                            'single_waveforms': json_data[session]['ppg']['single_waveforms']

                            #'single_waveforms_cr': get_change_ratio(data=block['ppg']['ppg45'], baseline=json_data[session_id]['rest']['ppg']['ppg45']),

                              })
                all_subject_data[subject] = feature_data
        #从所有数据里面分出训练集和测试集
        #这个人的所有数据作为测试集，其余人的数据作为训练集，
        for subject in all_subject_data:
            output_data = {
                'train': reduce(lambda feature_data_1, feature_data_2: merge(feature_data_1, feature_data_2), [all_subject_data[participant] for participant in all_subject_data if participant != subject]),
                'test': {
                    # '0': all_subject_data[subject]['0'][int(len(all_subject_data[subject]['0']) * TRAINING_DATA_RATIO):],
                    # '1': all_subject_data[subject]['1'][int(len(all_subject_data[subject]['1']) * TRAINING_DATA_RATIO):],
                    # '2': all_subject_data[subject]['2'][int(len(all_subject_data[subject]['2']) * TRAINING_DATA_RATIO):],
                    # '3': all_subject_data[subject]['3'][int(len(all_subject_data[subject]['3']) * TRAINING_DATA_RATIO):],
                    '0': all_subject_data[subject]['0'],
                    '1': all_subject_data[subject]['1'],
                    '2': all_subject_data[subject]['2'],
                    '3': all_subject_data[subject]['3'],
                },
            }

            dump_json(data=output_data, pathname=os.path.join(subject_independent_data_dir, '%s.json' % subject), overwrite=True)


if __name__ == '__main__':
    subject_independent()
