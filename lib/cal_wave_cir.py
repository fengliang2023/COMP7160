
import numpy as np
import scipy.signal as signal
from test_gen_audio import test_gen_audio
import matplotlib.pyplot as plt
import wave
import math

class cal_wave_cir():
    def __init__(self, waveData) -> None:
        obj = test_gen_audio(3/80)
        standard_wave = obj.wave_data
        # 下載波
        R_data = standard_wave
        Y_data = []
        fc = 20000
        fs = obj.sample_rate
        t = np.arange(0, len(R_data)/fs, 1/fs)  # 时间轴
        for i in range(R_data.size):
            # 计算复数部分
            angle = 2 * math.pi * fc * t[i]
            real = math.cos(angle) * math.sqrt(2)*R_data[i]
            Y_data.append(real)

        # 低通濾波
        Y_data = np.array(Y_data)
        reference_signal = Y_data
        fs = 48000
        wn1 = 2*4000/2/fs
        b, a = signal.butter(6,  wn1, 'lowpass')
        Y_data = signal.filtfilt(b,a,np.array(Y_data))
        # Y_data = Y_data.real
        standard_wave = Y_data / np.max(Y_data)
        # -------------------------------------------------------------------------------------------------------------

        # 下載波
        R_data = waveData
        Y_data = []
        fc = 20000
        fs = obj.sample_rate
        t = np.arange(0, len(R_data)/fs, 1/fs)  # 时间轴
        for i in range(R_data.size):
            # 计算复数部分
            angle = 2 * math.pi * fc * t[i]
            real = math.cos(angle) * math.sqrt(2)*R_data[i]
            # imag = - math.sin(angle) * math.sqrt(2)*R_data[i]
            # Y_data.append(complex(real, imag))
            Y_data.append(real)

        # 低通濾波
        Y_data = np.array(Y_data)
        reference_signal = Y_data
        fs = 48000
        wn1 = 2*4000/2/fs
        b, a = signal.butter(6,  wn1, 'lowpass')
        Y_data = signal.filtfilt(b,a,np.array(Y_data))
        # Y_data = Y_data.real
        Y_data = Y_data / np.max(Y_data)

        # 找最开始值
        tmp_data = Y_data[:600*10]
        start_index = np.argmax(np.correlate(tmp_data, standard_wave))
        # print("--------start_index--start_time------", start_index, start_index/48000)
        Y_data=Y_data[start_index:]

        frame_num = int(Y_data.size / 600)
        # print("--------frame_num--total_time------",frame_num, frame_num/80)
        Y_data = Y_data[:frame_num*600]

        # 下采樣
        # downsampe_data = []
        # for i in range(int(Y_data.size/12)):
        #     downsampe_data.append(Y_data[i*12])


        # # 计算cir
        # L=10
        # P=16
        # reseult_matrix = self.cal_cir_matrix(final_data[:312],L, P)
        # cir_list = []
        # for i in range(frame_num):
        #     start_index = i*600
        #     frame_data = Y_data[start_index:start_index+600]
        #     frame_data = self.down_sample(frame_data)
        #     RESULT = self.cal_cir(reseult_matrix, frame_data,L, P)
        #     cir_list.append(list(RESULT))


        # 计算cir
        L=140
        P=312 - L
        sequence = obj.LPF_data / np.max(np.abs(obj.LPF_data))
        reseult_matrix = self.cal_cir_matrix(obj.GSM_upsample[:312],L, P)
        # reseult_matrix = self.cal_cir_matrix(sequence,L, P)
        cir_list = []
        for i in range(frame_num):
            start_index = i*600
            frame_data = Y_data[start_index:start_index+600]
            RESULT = self.cal_cir(reseult_matrix, frame_data,L, P)
            # RESULT = RESULT / np.max(RESULT)
            cir_list.append(list(RESULT))

        CIRData = np.array(cir_list)
        self.CIRData= CIRData
        
    def main(self):
        return self.CIRData

    def cal_cir_matrix(self, origin_data, L, P):# reference length P(求解的参数个数); channel memory L(求解的cir值的个数)
        if P < L:
            print("警告：P < L，无法求解cir，自动修正为：P = L")
            P=L
        if P+L > len(origin_data) + 1:
            print("报错：P的值过大，训练序列过短，无法生成train_matrix")
            return 0
        train_matrix = []
        for i in range(P):
            train_matrix.append(list(origin_data[i:L + i]))    # 用切片方法时，最后一个数据取不到
        train_matrix = np.matrix(train_matrix)
        reseult_matrix = (train_matrix.T *train_matrix).I *  train_matrix.T
        return reseult_matrix

    def cal_cir(self, reseult_matrix,record_data, L, P=0):# reference length P; channel memory L
        record_data = np.array(record_data)
        record_matrix =  np.matrix(record_data[L:L+P]).T  # L-1:L+P-1]
        result = np.dot(reseult_matrix, record_matrix)
        return np.array(result).flatten()[::-1]
    
    def down_sample(self,data ):
        data = np.array(data)
        downsampe_data = []
        for i in range(int(data.size/12)):
            downsampe_data.append(data[i*12])
        return np.array(downsampe_data)