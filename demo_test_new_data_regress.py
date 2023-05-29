from __future__ import print_function

from ppg.utils import load_neural_network_model
from ppg.deep_learn import model_evaluate_regress

import os.path
import numpy as np
import os
import matplotlib.pyplot as plt
from ppg import BASE_DIR
from ppg.signal import smooth_ppg_signal, extract_ppg_single_waveform
from ppg.params import PPG_SAMPLE_RATE


def test_new_data(demo_data_raw):
    model_dir = os.path.join(BASE_DIR, 'models', 'regress')

    #来了两分钟的数据，首先把数据预处理，切成一个波形一个波形的
    #然后输入模型分类就可以了，转化为list形式
    #demo_data_raw是两分钟原始数据，
    #数据切割为一个一个波形
    demo_data_waveforms = extract_ppg_single_waveform(\
        signal=smooth_ppg_signal(signal=demo_data_raw, sample_rate=PPG_SAMPLE_RATE))

    # loaded model，取model
    model_path = r'周逸菲'
    pathname_model = os.path.join(model_dir, model_path)
    print(pathname_model)
    loaded_model = load_neural_network_model(pathname=pathname_model)
    #ynew,预测的种类值
    ynew = model_evaluate_regress(loaded_model, demo_data_waveforms)
    plt.plot(ynew)
    plt.show()
    mean_workload_all = np.mean(ynew)
    return mean_workload_all


if __name__ == '__main__':
    demo_data_raw = []  #数据类型为list
    #读取本地数据，模拟实时传输过来的数据
    pathname = 'BVP_n-back.csv'
    if os.path.exists(pathname):
        with open(pathname) as f:
            lines = f.readlines()
            for line in lines[1:]:
                demo_data_raw.append(int(float(line.strip())))  # BVP的每一行数据存入数组中
    #进行压力预测
    predicted_workload = test_new_data(demo_data_raw)
    print(predicted_workload)
