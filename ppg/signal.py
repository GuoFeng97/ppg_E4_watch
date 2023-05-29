# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import argrelmax, argrelmin, firwin, convolve
from scipy.interpolate import interp1d
from .params import MINIMUM_PULSE_CYCLE, MAXIMUM_PULSE_CYCLE
from .params import PPG_SAMPLE_RATE, PPG_FIR_FILTER_TAP_NUM, PPG_FILTER_CUTOFF, PPG_SYSTOLIC_PEAK_DETECTION_THRESHOLD_COEFFICIENT
from .params import ECG_R_PEAK_DETECTION_THRESHOLD
import matplotlib.pyplot as plt

def find_extrema(signal):
    signal = np.array(signal)
    extrema_index = np.sort(np.unique(np.concatenate((argrelmax(signal)[0], argrelmin(signal)[0]))))
    extrema = signal[extrema_index]
    return zip(extrema_index.tolist(), extrema.tolist())


def smooth_ppg_signal(signal, sample_rate=PPG_SAMPLE_RATE, numtaps=PPG_FIR_FILTER_TAP_NUM, cutoff=PPG_FILTER_CUTOFF):
    if numtaps % 2 == 0:
        numtaps += 1
    return convolve(signal, firwin(numtaps, [x*2/sample_rate for x in cutoff], pass_zero=False), mode='valid').tolist()


def validate_ppg_single_waveform(single_waveform, sample_rate=PPG_SAMPLE_RATE):
    period = len(single_waveform) / sample_rate

    # plt.plot(single_waveform)
    # plt.show()

    if period < MINIMUM_PULSE_CYCLE or period > MAXIMUM_PULSE_CYCLE:
        return False
    max_index = np.argmax(single_waveform)
    if max_index / len(single_waveform) >= 0.5:
        return False
    # temp = argrelmax(np.array(single_waveform))
    # temp1 = len(temp[0])
    # if len(argrelmax(np.array(single_waveform))[0]) < 2:
    #     return False
    min_index = np.argmin(single_waveform)
    if not (min_index == 0 or min_index == len(single_waveform) - 1):
        return False
    diff = np.diff(single_waveform[:max_index+1], n=1)
    if min(diff) < 0:
        return False
    if abs(single_waveform[0] - single_waveform[-1]) / (single_waveform[max_index] - single_waveform[min_index]) > 0.1:
        return False
    return True


def extract_ppg_single_waveform(signal, sample_rate=PPG_SAMPLE_RATE):

    y_filted = np.array(signal)

    plt.plot(signal)
    plt.show()

    #y_filted = signal
    #把整段数据切割成一个一个的波形
    #used 用于标识每一次切割,未切割下来的波形处置为1，切割下来的为0
    initial_value = 1
    list_length = len(y_filted)
    used = [initial_value] * list_length
    used = np.array(used)
    #used(1: 300)=0; % 前300个点舍弃（归零）
    SpacePoint = [] #; % 存储分割点位置
    length_y_filted= len(y_filted)

    threshold = 0

    while min(used * y_filted) < -0.1:  # 如果波形的最低点 < -0.1, 或者-1, 或者10，每个人不同
        tmp = used * y_filted
        pos = np.where(tmp == np.min(tmp))[0][0]   # mini是最低点，若有多个最低点，pos是第一个最低点位置
        SpacePoint.append(pos)  # 记录最低点位置
        tmp_pos = pos #; % tmp_pos记录下找到的第一个最低点的位置
        used[tmp_pos] = 0#; % 将原始波形里的找到的第一个最低点归零
        #% 找到最低点后面第一个波峰值，如果超出矩阵维度就让threshold = 0；
        if (tmp_pos + sample_rate) < length_y_filted:
            first_top = max(y_filted[tmp_pos:(tmp_pos + sample_rate)])
            threshold = first_top - (first_top - y_filted[tmp_pos]) / 4

        while (y_filted[tmp_pos] < threshold and tmp_pos > 0 and (pos - tmp_pos) < sample_rate):
            tmp_pos = tmp_pos - 1
            used[tmp_pos] = 0 #; % 将最低点前(一个周期内)
            #所有小于阈值threshold的点都归零，只要正值

        tmp_pos = pos
        while (y_filted[tmp_pos] < threshold and tmp_pos < len(y_filted)-1 and (tmp_pos - pos) < sample_rate):
            tmp_pos = tmp_pos + 1
            used[tmp_pos] = 0  #将最低点后(一个周期内), 所有小于阈值threshold的点都归零，只要正值

    SpacePoint = np.array(SpacePoint)
    SpacePoint.sort(axis=0) #% % % % % % 各个区间的分割点，对SpacePoint中元素按照升序排列

    single_waveforms = []
    for i in range(0, len(SpacePoint)-2):
        single_waveform = y_filted[SpacePoint[i]:SpacePoint[i + 1]]  #  分割单周期

        # plt.plot(single_waveform)
        # plt.show()

        if validate_ppg_single_waveform(single_waveform=single_waveform, sample_rate=sample_rate):
            single_waveform = single_waveform.tolist()

            single_waveforms.append(single_waveform)


    return single_waveforms



def extract_rri(signal, sample_rate):
    rri = []
    rri_time = []
    last_extremum_index = None
    last_extremum = None
    last_r_peak_index = None
    for extremum_index, extremum in find_extrema(signal=signal):
        if last_extremum is not None and extremum - last_extremum > ECG_R_PEAK_DETECTION_THRESHOLD:
            if last_r_peak_index is not None:
                interval = (extremum_index - last_r_peak_index) / sample_rate
                if interval >= MINIMUM_PULSE_CYCLE and interval <= MAXIMUM_PULSE_CYCLE:
                    rri.append(interval)
                    rri_time.append(extremum_index / sample_rate)
            last_r_peak_index = extremum_index
        last_extremum_index = extremum_index
        last_extremum = extremum
    return rri, rri_time


def interpolate_rri(rri, rri_time, sample_rate):
    f = interp1d(rri_time, rri, kind='cubic')
    step = 1 / sample_rate
    return f(np.arange(rri_time[0], rri_time[-1] - step, step)).tolist()