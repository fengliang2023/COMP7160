# 导入所需模块
# import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import BatchNormalization
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.initializers import TruncatedNormal
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dropout
# from keras.layers.core import Dense
from keras import backend as K
# kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05)

class SimpleVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# 不同工具包颜色通道位置可能不一致
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL 
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL 
		model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# (CONV => RELU) * 3 => POOL 
		model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# FC层
		model.add(Flatten())
		model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.6))

		# softmax 分类
		model.add(Dense(classes,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
		model.add(Activation("softmax"))

		return model