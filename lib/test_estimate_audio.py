import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
# backup_folder = os.path.join(current_folder, '..', 'lib')
# sys.path.append(backup_folder)
from cal_wave_cir import cal_wave_cir
from test_gen_audio import test_gen_audio

import numpy as np
import wave
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import shutil
import datetime
# import pandas as pd


# _, _, files = next(os.walk(current_folder))
# wav_files = [file for file in files if file.endswith('.wav')]
# print(wav_files)

class test_estimate_audio():
    def __init__(self, para_dic) -> None:
        self.wave_file_dir =para_dic["save_file_dir"]    # 原始音频的目录
        self.gesture_list = [para_dic["estimate_audio_file"]]
        self.start_input = para_dic["duration_noise"]    # 整段音频的起始时间（单位为秒）
        self.len_each_chip = para_dic["len_each_segment"]    # 单个手势chip的总时间（单位s为秒）S
        self.start_time_chip = para_dic["start_time_CIR"]    # 计算cir从第几秒开始g（cir图的起始时间点）
        self.len_time_chip = para_dic["duration_CIR"]    # 计算cir的持续几秒（cir图的长度）

    def main(self):
        for wave_file in self.gesture_list:
            self.init_num = 0  # 文件命名的开始序号
            gest_name = wave_file.split(".wav")[0]
            dir_path =  "CIR/" + gest_name
            if os.path.exists(dir_path):
                # print("The current directory exists. Please save the directory data first:  ",dir_path)
                # continue
                pass
            else:
                os.makedirs(dir_path)# 如果不存在，创建新目录

            input_filename = self.wave_file_dir + wave_file
            self.cal_cir(input_filename, gest_name)
            print("--------saving_file-------", dir_path)

    def cal_cir(self, input_filename, gest_name):
        # 读取数据
        f = wave.open(input_filename, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)#读取音频，字符串格式
        waveData = np.frombuffer(strData,dtype=np.int16)#将字符串转化为int
        nframes = len(waveData) // nchannels
        waveData = np.reshape(waveData,[nframes,nchannels]).T
        waveData = waveData[0,:] #只取channels=0时的数据
        f.close()
        waveData = waveData[int(framerate*self.start_input):]
        self.num_chip = int(len(waveData)*1.0 / (framerate*self.len_each_chip))
        print("-------self.num_chip----",self.num_chip)

        for i in range(self.num_chip):
            start_index = framerate*(self.start_time_chip + i*self.len_each_chip)
            end_index = start_index + framerate*self.len_time_chip + 600*10   # 输入5s，由于需要定位初始点，所以结果会少于5*80帧数据，所以再补上10帧的数据
            AUDIO_DATA = waveData[start_index:end_index]
            CIRData, diffCIR = self.cal_CIR_dCIR(AUDIO_DATA, 80*self.len_time_chip)
            # 保存数据
            gestr = input_filename
            save_file_name = "CIR/" + gest_name + "/" + gest_name + "_{:04d}".format(i + self.init_num)
            # print("--------saving_file-------", save_file_name)
            self.save_to_csv(CIRData, diffCIR, save_file_name)

    def cal_CIR_dCIR(self, AUDIO_DATA, len_cir):
        # 计算cir
        obj = cal_wave_cir(AUDIO_DATA)
        CIRData = obj.main()
        CIRData = CIRData[:len_cir, :]
        # 计算dcir
        diffCIR = np.zeros_like(CIRData)
        for j in range(1,CIRData.shape[0]):
            diffCIR[j,:] =CIRData[j,:] - CIRData[j-1,:]
        # 归一化
        new_mean = np.mean(diffCIR)
        new_mean = np.mean(diffCIR[diffCIR > new_mean])
        # new_mean = np.mean(diffCIR[diffCIR > new_mean])
        end_value = np.mean(diffCIR[diffCIR > new_mean])
        diffCIR = np.interp(diffCIR, (diffCIR.mean(), end_value), (0, 1))
        CIRData = np.interp(CIRData, (CIRData.min(), CIRData.max()), (0, 1))
        return CIRData, diffCIR

    def save_to_csv(self, CIRData, diffCIR, filename):
        fig, ax = plt.subplots()
        ax.imshow(diffCIR.T)
        ax.axis('off')  # 关闭坐标轴
        plt.savefig(filename+'.png', dpi=300, bbox_inches='tight', pad_inches=0)  # 保存图片，设置分辨率为300dpi，并且裁剪空白区域s
        plt.close()
