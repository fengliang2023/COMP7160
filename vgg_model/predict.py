# 导入所需工具包
from keras.models import load_model
import argparse
import pickle
import cv2
import random
import os

directory = 'audio_phase'# 目录路径
subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]# 获取目录下所有子目录的路径
random_subdirectory = random.choice(subdirectories)# 随机选择一个子目录
files = [os.path.join(random_subdirectory, f) for f in os.listdir(random_subdirectory) if os.path.isfile(os.path.join(random_subdirectory, f))]# 获取该子目录下所有文件的路径
random_file = random.choice(files)# 随机选择一个文件
print("---------------测试的图片：----------------",random_file)
# exit()

args = {}
args["image"] = random_file
args["model"] = "output_cnn"
args["label_bin"] = "output_cnn/vggnet_lb.pickle"
args["width"] = 64
args["height"] = 64
args["flatten"] = -1


# 加载测试数据并进行相同预处理操作
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# 是否要对图像就行拉平操作(3维变1维)
if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# 读取模型和标签
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# 预测
preds = model.predict(image)
# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# 在图像中把结果画出来
text1 = "Origin:   {}".format(random_file.split("\\")[1])
text2 = "Result:   {}  {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(output, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 绘图
cv2.imshow("Image", output)
cv2.waitKey(0)