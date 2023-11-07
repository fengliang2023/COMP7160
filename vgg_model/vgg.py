# 导入所需工具包
import matplotlib
from model_name.simple_vggnet import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from my_utils import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.layers import Activation

# 设置参数
args = {}
args["dataset"] = "CIR"
args["model"] = "output_cnn"
args["label_bin"] = "output_cnn/vggnet_lb.pickle"
args["plot"] = "output_cnn"

figure_width=64
figure_height=64
figure_depth=3

# 读取数据和标签
print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(utils_paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)    # 将文件名打乱
# print(imagePaths)
# exit()

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (figure_width, figure_height))
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# 预处理
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# 标签转换
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# print(lb.classes_)
# exit()
# 数据增强
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# 建立卷积神经网络
model = SimpleVGGNet.build(width=figure_width, height=figure_height, depth=figure_depth,
	classes=len(lb.classes_))

# 初始化超参数
INIT_LR = 0.01
EPOCHS = 300  # 300  #30  #1000
BS = 32 #64  #32

# 损失函数
print("[INFO] 训练网络...")
# opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 训练网络
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=BS)
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS)

# 测试
print("[INFO] 测试网络...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# 展示结果
# N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["acc"], label="train_acc")
# plt.plot(N, H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy (SmallVGGNet)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig(args["plot"])

# 保存模型
print("[INFO] 保存模型...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()