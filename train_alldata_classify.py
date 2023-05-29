# -*- coding: utf-8 -*-

import os
import fnmatch
from ppg import BASE_DIR
from ppg.utils import exist, load_json, dump_json, load_model, dump_model, export_csv,save_neural_network_model
from ppg.learn import get_feature_set
from ppg.learn_new import get_feature_set_waveforms
from ppg.learn import logistic_regression_classifier
from ppg.learn import support_vector_classifier
from ppg.learn import gaussian_naive_bayes_classifier
from ppg.learn import decision_tree_classifier
from ppg.learn import random_forest_classifier, adaboost_classifier, gradient_boosting_classifier
from ppg.learn import voting_classifier
from ppg.deep_learn import deep_learning,deep_learning_regress


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

def classify():
    feature_data_dir = os.path.join(BASE_DIR, 'data', 'make_all_data_train_calculate_rest_5min')
    model_dir = os.path.join(BASE_DIR, 'models', 'classify_all')
    result_dir = os.path.join(BASE_DIR, 'results', 'waveforms')

    level_sets = [
        # ['0', '2'],
        ['0', '1','3']
        #['1', '3'],
    ]
    #  0 rest 1 hard 2 rest 3 easy


    feature_type_sets = [
        ['single_waveforms']
    ]
    classifiers = [
        #('deep_learning', deep_learning, ),
        #('support_vector', support_vector_classifier, ),
        # ('gaussian_naive_bayes', gaussian_naive_bayes_classifier, ),
        # ('decision_tree', decision_tree_classifier, ),
    ]

    if exist(pathname=feature_data_dir):
        result_data = {}
        for filename_with_ext in fnmatch.filter(os.listdir(feature_data_dir), '*.json'):
            participant = os.path.splitext(filename_with_ext)[0]
            pathname = os.path.join(feature_data_dir, filename_with_ext)
            json_data = load_json(pathname=pathname)
            if json_data is not None:
                for level_set in level_sets:
                    level_set_name = '-'.join(level_set)
                    if level_set_name not in result_data:
                        result_data[level_set_name] = {}
                    for feature_type_set in feature_type_sets:
                        feature_type_set_name = '-'.join(feature_type_set)
                        if feature_type_set_name not in result_data[level_set_name]:
                            result_data[level_set_name][feature_type_set_name] = {}
                        # from JSON to list
                        #采用leave one out 来求
                        #训练数据，是除了这个人外所有人的
                        x_train, y_train, x_test, y_test = get_feature_set_waveforms(data=json_data, level_set=level_set, feature_type_set=feature_type_set)
                        estimators = []
                        #采用神经网络
                        # classifier, score = deep_learning(x_train, y_train, x_test, y_test)
                        # 0 rest=0, 1 hard = 0.9, 3 easy=0.3，需要改函数里面
                        classifier,score = deep_learning(x_train, y_train, x_test, y_test)
                        # 把模型保存起来
                        classifier_name = "deep_learning"
                        # serialize weights to HDF5
                        model_pathname = os.path.join(model_dir, level_set_name, feature_type_set_name, classifier_name,
                                                      participant)
                        save_neural_network_model(model=classifier, pathname=model_pathname)


if __name__ == '__main__':
    # classify(feature_data='splited')
    classify( )
