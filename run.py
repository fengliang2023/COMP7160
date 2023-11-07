
import sys
sys.path.append('./lib')  # 将lib目录添加到Python模块搜索路径中
from test_gen_audio import test_gen_audio 
from test_estimate_audio import test_estimate_audio


gen_audio = 0  # generate detection signal

estimate_audio = 1  # estimate the CIR

if __name__ == "__main__":
    if gen_audio:
        signal_duration = 60    # the duration of detection signal(in seconds)
        obj = test_gen_audio(signal_duration)
        obj.main()
    if estimate_audio:
        para_dic = {
        "estimate_audio_file": "test4.wav",    # The filename of estimated audio file
        "save_file_dir" : "./",    # The directory for results
        "duration_noise": 5,    # The length of the initial noise to removed(in seconds)
        "len_each_segment": 4,    # The total length of each segment sample to estimate CIR image (in seconds)
        "start_time_CIR": 2,    # The starting time to estimate CIR image in the first segment sample(in seconds)
        "duration_CIR": 2    # The length of CIR image (in seconds)
        }
        obj_cal_cir = test_estimate_audio(para_dic)
        obj_cal_cir.main()

