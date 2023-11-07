
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
import wave
import datetime

class test_gen_audio():
    def __init__(self, signal_duration=1) -> None:
        num = int(80*signal_duration)    # num=1约等于600/48000=1/80s
        # print("----",num)
        self.bandwidth = 4000
        self.sample_rate = 48000
        GSM_data =  [-1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1]    # 26个
        final_data = GSM_data + [-1]*24
        self.GSM_pad = final_data

        # 上采樣
        result_list = []
        for i in range(len(final_data)):
            result_list.extend([final_data[i]]*12)

        result_list = result_list*num
        self.GSM_upsample = result_list

        # 低通濾波
        w = self.bandwidth
        fs = self.sample_rate
        wn1 = 2*w/2/fs
        b, a = signal.butter(6,  wn1, 'lowpass')
        S_data = signal.filtfilt(b,a,np.array(result_list))
        self.LPF_data = S_data

        # 上载波
        fc = 20000
        t = np.arange(0, len(S_data)/fs, 1/fs)  # 时间轴
        S_data = S_data* math.sqrt(2) * np.cos(2*np.pi*fc*t)

        # 通带滤波
        fs = self.sample_rate
        wn1 = 2*18000/fs
        wn2 = 2*22000/fs
        b, a = signal.butter(6,  [wn1,wn2], 'bandpass')
        S_data = signal.filtfilt(b,a,np.array(S_data))
        self.wave_data = S_data

    def main(self):
        y = self.wave_data *10000
        fs = self.sample_rate
        data = datetime.datetime.now().date()
        audio_filename = "gen_audio_" + str(data) + ".wav"
        y =  np.array(y)
        # print("-------wave_data--------",np.max(np.abs(y)),y[:5])
        # print("-----framerate-----:",fs)
        print("-----audio_filename-----:",audio_filename)
        f = wave.open(audio_filename, "wb")
        f.setnchannels(1)           # 配置声道数
        f.setsampwidth(2)           # 配置量化位数
        f.setframerate(fs)   # 配置取样频率
        f.writeframes(y.astype(np.short).tobytes())
        f.close()

