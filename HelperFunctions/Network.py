import numpy as np
import cv2
import os
import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras.utils
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Input, Dense, Activation, \
	ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Model, load_model

K.set_image_data_format('channels_last')


class ConvolutionalNeuralNetwork:
	def __init__(self, input_shape=(16, 32, 3), output_size=10, model_type="LeNet5", learning_rate=0.001, loadModel=True):
		self.model = None
		self.loadModel = loadModel
		self.learning_rate = learning_rate
		self.input_shape = input_shape
		self.reshape_size = (32, 64, 3)
		self.output_size = output_size
		self.model_type = model_type
		self.NumberSizes = {1}
		self.setup()
	
	def calc_overlap_area(self, a, b):
		# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles/27162334#27162334
		# a = 'xmin ymin xmax ymax')
		change_in_x = min(a[2], b[2]) - max(a[0], b[0])
		change_in_y = min(a[3], b[3]) - max(a[1], b[1])
		overlap_area = 0
		if (change_in_x >= 0) and (change_in_y >= 0):
			overlap_area = change_in_x * change_in_y
		# Modified output to return largest percentage of overlap WRT each bounding box
		return max(overlap_area / (a[4] * a[5]), overlap_area / (b[4] * b[5]))
	
	def calc_likelihood(self, bboxes, confidence, predictions):
		bboxes = np.asarray(bboxes)
		confidence = np.asarray(confidence)
		all_bb = []
		all_pred = []
		all_conf = []
		for i in range(len(bboxes)):
			counts = np.unique(np.asarray(predictions[i]), return_counts=True)
			new_bbox = []
			new_pred = []
			new_conf = []
			for tc in counts[0]:
				t = np.asarray(bboxes[i])[predictions[i] == tc]
				t_conf = confidence[i][predictions[i] == tc]
				size = t[:, 2] * t[:, 3]
				max_idx = np.argmax(size, axis=0)
				new_bbox.append(t[max_idx])
				new_pred.append(tc)
				new_conf.append(t_conf[max_idx])
			all_bb.append(np.asarray(new_bbox))
			all_pred.append(np.asarray(new_pred))
			all_conf.append(np.asarray(new_conf))
		return np.asarray(all_bb), np.asarray(all_pred), np.asarray(all_conf)
	
	def load_vgg(self, use_weights=False):
		modelPath = os.getcwd() + "\\Data\\TrainedModels\\"
		if use_weights:
			model_type = "VGG16"
			reshape_size = (32, 64, 3)
			learn_rate = 0.00005
			model = ConvolutionalNeuralNetwork(input_shape=reshape_size, model_type=model_type,
			                                   output_size=11, learning_rate=learn_rate)
			model.model.load_weights("vgg_weights.h5")
			return model.model
		else:
			return load_model(modelPath + "Trained_Model_VGG16_Acc_0.957_Sess_49.h5")
		
	def processImage(self, processedImage):
		original = np.copy(processedImage)
		# Check if color image or not
		if len(processedImage.shape) == 3:
			# It is a Color Image
			processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)
		
		processedImage[processedImage < 75] = 0
		processedImage[processedImage >= 75] = 255
		edges = cv2.Canny(processedImage, threshold1=75, threshold2=125, apertureSize=3)
		# Contours read in reverse order
		contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(i) for i in contours]

		return original, processedImage
		
	def predict(self, img):
		originalImage, processedImage = self.processImage(processedImage=img)
		
		return
	
	def evaluate(self):
		return
	
	def create_convolutional_segment(self, X, mid_conv_window, filters, name, name_pt_2, stride=2, add_dropout=False):
		convolutional_layer_base = 'res' + str(name) + name_pt_2 + '_branch'
		batch_norm_layer_base = 'bn' + str(name) + name_pt_2 + '_branch'
		if len(filters) == 1:
			F1 = filters
		elif len(filters) == 2:
			F1, F2 = filters
		elif len(filters) == 3:
			F1, F2, F3 = filters
		X = Activation('relu')(X)
		X_shortcut = X
		if len(filters) == 1:
			X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride), name=convolutional_layer_base + '2a',
			           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2a')(X)
			if add_dropout:
				X = Dropout(0.1)(X)
			X_shortcut = Conv2D(filters=F1, kernel_size=(1, 1), strides=(stride, stride),
			                    name=convolutional_layer_base + '1', padding='valid',
			                    kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
			X_shortcut = BatchNormalization(axis=3, name=batch_norm_layer_base + '1')(X_shortcut)
		if len(filters) == 2:
			X = Conv2D(filters=F2, kernel_size=(mid_conv_window, mid_conv_window), strides=(1, 1),
			           name=convolutional_layer_base + '2b', padding='same',
			           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2b')(X)
			if add_dropout:
				X = Dropout(0.1)(X)
			X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(stride, stride),
			                    name=convolutional_layer_base + '1', padding='valid',
			                    kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
			X_shortcut = BatchNormalization(axis=3, name=batch_norm_layer_base + '1')(X_shortcut)
		if len(filters) == 3:
			X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=convolutional_layer_base + '2c',
			           padding='valid',
			           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			X = BatchNormalization(axis=3, name=batch_norm_layer_base + '2c')(X)
			X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(stride, stride),
			                    name=convolutional_layer_base + '1', padding='valid',
			                    kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
			X_shortcut = BatchNormalization(axis=3, name=batch_norm_layer_base + '1')(X_shortcut)
		X = tensorflow.keras.layers.Add()([X, X_shortcut])
		X = Activation('relu')(X)
		return X
	
	def create_identity_segment(self, X, mid_conv_window, filters, name, name_pt_2):
		conv_name_base = 'res' + str(name) + name_pt_2 + '_branch'
		bn_name_base = 'bn' + str(name) + name_pt_2 + '_branch'
		F1, F2, F3 = filters
		X_shortcut = X
		
		X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F2, kernel_size=(mid_conv_window, mid_conv_window),
		           strides=(1, 1), padding='same', name=conv_name_base + '2b',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
		X = Activation('relu')(X)
		X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
		           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
		X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
		
		X = tensorflow.keras.layers.Add()([X_shortcut, X])
		X = Activation('relu')(X)
		return X
	
	def create_model(self, input_shape, model_type="LeNet5", addDropOut=False):
		if model_type == "LeNet5" or model_type.lower() == "lenet5" or model_type.lower() == "lenet":
			X_in = Input(input_shape)
			
			# X = ZeroPadding2D((3, 3))(X_in)
			#
			# num_filters_conv1 = 4
			# X = Conv2D(num_filters_conv1, (3, 3), strides=(1, 1), name='convolution_zero')(X)
			# X = BatchNormalization(axis=3, name='batch_normalization_zero')(X)
			# X = AveragePooling2D((2, 2), name='max_pool_zero')(X)
			# X = Activation('relu')(X_in)
			filter_1 = [3, 6]
			X = self.create_convolutional_segment(X_in, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='a',
			                                      stride=1)
			if addDropOut:
				X = Dropout(0.2)(X)
			
			# num_filters_conv2 = 4
			# X = Conv2D(num_filters_conv2, (3, 3), strides=(1, 1), name='convolution_one')(X)
			# X = BatchNormalization(axis=3, name='batch_normalization_one')(X)
			# X = AveragePooling2D((2, 2), name='max_pool_one')(X)
			# X = Activation('relu')(X)
			#
			# X = Flatten()(X)
			# X = Dense(1048, activation='sigmoid', name='fully_connected_zero')(X)
			# if addDropOut:
			# 	X = Dropout(0.5)(X)
			#
			# X = Flatten()(X)
			# X = Dense(512, activation='sigmoid', name='fully_connected_one')(X)
			# if addDropOut:
			# 	X = Dropout(0.5)(X)
			#
			X = Flatten()(X)
			X = Dense(64, activation='sigmoid', name='fully_connected_three')(X)
			
			X = Flatten()(X)
			X = Dense(self.output_size, activation='softmax', name='fully_connected_four')(X)
			
			model = Model(inputs=X_in, outputs=X, name='LeNet5')
			return model
		elif model_type == "VGG16" or model_type.lower() == "vgg16":
			vgg = tensorflow.keras.applications.VGG16(include_top=False, weights='imagenet',
			                                          input_shape=self.input_shape)
			for layer in vgg.layers[:-1]:
				layer.trainable = True
			X = Flatten()(vgg.output)
			# filter_1 = [64, 64, 256]
			# X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1,
			#                                       name=2, name_pt_2='a', stride=1, add_dropout=True)
			X = Dense(self.output_size, activation='softmax')(X)
			model = Model(inputs=vgg.input, outputs=X)
			return model
		elif model_type == "ResNet" or model_type.lower() == "resnet":
			X_in = Input(input_shape)
			X = ZeroPadding2D((3, 3))(X_in)
			
			X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
			           kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			X = BatchNormalization(axis=3, name='bn_conv1')(X)
			X = Activation('relu')(X)
			X = MaxPooling2D((3, 3), strides=(2, 2))(X)
			
			filter_1 = [64, 64, 256]
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='a',
			                                      stride=1)
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_1, name=2, name_pt_2='b',
			                                      stride=1)
			X = self.create_identity_segment(X, 3, filter_1, name=2, name_pt_2='c')
			X = self.create_identity_segment(X, 3, filter_1, name=2, name_pt_2='d')
			
			filter_2 = [128, 128, 512]
			X = self.create_convolutional_segment(X, mid_conv_window=3, filters=filter_2, name=3, name_pt_2='a',
			                                      stride=2)
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='b')
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='c')
			X = self.create_identity_segment(X, 3, filter_2, name=3, name_pt_2='d')
			
			X = AveragePooling2D((2, 2), name="avg_pool")(X)
			
			X = Flatten()(X)
			X = Dense(self.output_size, activation='softmax', name='fc' + str(self.output_size),
			          kernel_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0))(X)
			
			model = Model(inputs=X_in, outputs=X, name='ResNet50')
			return model
	
	def setup(self):
		if self.loadModel:
			self.model = self.load_vgg()
		else:
			self.model = self.create_model(self.input_shape, model_type=self.model_type)
			self.model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate),
			                   loss=tensorflow.keras.losses.categorical_crossentropy, metrics=["accuracy"])
			return
