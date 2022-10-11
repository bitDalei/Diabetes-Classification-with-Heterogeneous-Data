import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.keras.backend import concatenate

from tensorflow.python.ops.gen_array_ops import concat 

from utils.utils import save_logs
from utils.utils import calculate_metrics

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier_RESNETMLP:

	def __init__(self, output_directory, inputA_shape, inputB_shape, nb_classes, verbose=False,build=True,\
		nb_filters=32, use_residual=True, use_bottleneck=True, depth=3, kernel_size=41,flag_att=0,flag_CBAM=0):
		self.output_directory = output_directory

		self.flag_CBAM = flag_CBAM
		self.flag_att = flag_att

		if build == True:
			self.model = self.build_model(inputA_shape, inputB_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			# self.model.save_weights(self.output_directory+'model_init.hdf5')
			self.methodName = "resnetMLP"
		return

	def _inception_module(self, input_tensor, stride=1, activation='linear'):

		if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
			input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
													padding='same', activation=activation, use_bias=False)(input_tensor)
		else:
			input_inception = input_tensor

		# kernel_size_s = [3, 5, 8, 11, 17]
		kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

		conv_list = []

		for i in range(len(kernel_size_s)):
			conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
													strides=stride, padding='same', activation=activation, use_bias=False)(
				input_inception))

		max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

		conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
										padding='same', activation=activation, use_bias=False)(max_pool_1)

		conv_list.append(conv_6)

		x = keras.layers.Concatenate(axis=2)(conv_list)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation(activation='relu')(x)
		return x

	def _shortcut_layer(self, input_tensor, out_tensor):
		shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
											padding='same', use_bias=False)(input_tensor)
		shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

		x = keras.layers.Add()([shortcut_y, out_tensor])
		x = keras.layers.Activation('relu')(x)
		return x

	def channel_attention(self, input_feature, ratio=8):
		channel = input_feature.shape[-1]

		shared_layer_one = keras.layers.Dense(channel//ratio,kernel_initializer='he_normal',activation = 'relu',use_bias=True,bias_initializer='zeros')
		shared_layer_two = keras.layers.Dense(channel,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
		
		avg_pool = keras.layers.GlobalAveragePooling1D()(input_feature)  
		avg_pool = shared_layer_one(avg_pool)
		avg_pool = shared_layer_two(avg_pool)
		
		max_pool = keras.layers.GlobalMaxPooling1D()(input_feature)
		max_pool = shared_layer_one(max_pool)
		max_pool = shared_layer_two(max_pool)
		
		cbam_feature = keras.layers.Add()([avg_pool,max_pool])
		cbam_feature = keras.layers.Activation('hard_sigmoid')(cbam_feature)
		return keras.layers.multiply([input_feature, cbam_feature])

	def spatial_attention(self, input_feature):
		kernel_size = 3
		cbam_feature = input_feature
		
		avg_pool = keras.layers.Lambda(lambda x: keras.backend.mean(x, keepdims=True))(cbam_feature)
		max_pool = keras.layers.Lambda(lambda x: keras.backend.max(x, keepdims=True))(cbam_feature)
		concat = concatenate([avg_pool, max_pool])
		cbam_feature = keras.layers.Conv1D(filters = 1,kernel_size=kernel_size,activation = 'hard_sigmoid',\
			strides=1,padding='same',kernel_initializer='he_normal',use_bias=False)(concat)
		
		return keras.layers.multiply([input_feature, cbam_feature])

	def cbam_block(self, cbam_feature,ratio=8):
		"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
		As described in CBAM: Convolutional Block Attention Module.
		"""
		cbam_feature = self.channel_attention(cbam_feature, ratio)
		cbam_feature = self.spatial_attention(cbam_feature, )
		return cbam_feature

	def build_model(self, inputA_shape, inputB_shape, nb_classes, n_feature_maps=64):
		inputA_layer = keras.layers.Input(inputA_shape)
		inputB_layer = keras.layers.Input(inputB_shape)
        
		# BLOCK 1
		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(inputA_layer)
		conv_x = keras.layers.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.BatchNormalization()(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(inputA_layer)
		shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
		
		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv_z)
			output_block_1 = keras.layers.add([shortcut_y, conv_z, cbam])
		else:
			output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2

		conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.BatchNormalization()(conv_z)

		# expand channels for the sum
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv_z)
			output_block_2 = keras.layers.add([shortcut_y, conv_z, cbam])
		else:
			output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3

		conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal
		shortcut_y = keras.layers.BatchNormalization()(output_block_2)

		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv_z)
			output_block_3 = keras.layers.add([shortcut_y, conv_z, cbam])
		else:
			output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL

		pooling_layerA = keras.layers.GlobalAveragePooling1D()(output_block_3)
		dense_layerA = keras.layers.Dense(64, activation='softmax')(pooling_layerA)

		modelA = keras.models.Model(inputs=inputA_layer, outputs=dense_layerA)

		# B part, biomarkers
		input_layer_flattened = keras.layers.Flatten()(inputB_layer)

		layer_1 = keras.layers.Dropout(0.2)(input_layer_flattened)
		layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

		layer_2 = keras.layers.Dropout(0.2)(layer_1)
		layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

		output_layerB = keras.layers.Dropout(0.3)(layer_2)
		output_layerB = keras.layers.Dense(64, activation='relu')(output_layerB)

		modelB = keras.models.Model(inputs=inputB_layer, outputs=output_layerB)

		# combine
		combined = concatenate([modelA.output, modelB.output])
		combined = tf.expand_dims(combined, -1)
		
		if self.flag_att == 1:
			weight = keras.layers.Dense(128, activation='softmax')(combined)
			combined = weight * combined
			combined = keras.layers.Conv1D(128, kernel_size=8, padding='same')(combined)
		conv1 = keras.layers.BatchNormalization()(combined)
		conv1 = keras.layers.Activation('relu')(conv1)

		pooling_layer = keras.layers.GlobalAveragePooling1D()(conv1)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(pooling_layer)

		model = keras.models.Model(inputs=[modelA.input, modelB.input], outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=1e-3), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=1e-8)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 


	def fit(self, x_trainAB, y_train, x_valAB, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 32
		nb_epochs = 800

		start_time = time.time() 

		hist = self.model.fit(x_trainAB, y_train, batch_size=batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_valAB,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_valAB)
		np.save(self.output_directory + self.methodName + '_y_pred.npy', y_pred)
		np.save(self.output_directory + self.methodName + '_y_true.npy', y_true)
		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_testAB, y_true, return_df_metrics = False):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_testAB)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred

	def evaluate(self):
		pred = np.load(self.output_directory + self.methodName +'_y_pred.npy')
		true = np.load(self.output_directory + self.methodName +'_y_true.npy')

		# print(self.methodName + " start")

		predList = []
		trueList = []

		cnt = 0
		for i in range(len(pred)):

			a = pred[i].tolist()
			b = true[i]

			ida = a.index(max(a)) + 1

			predList.append(ida)
			trueList.append(b)

			if ida == b:
				cnt += 1

		# print('accuracy:', cnt/len(pred))

		sns.set()
		f,ax=plt.subplots()
		C2= confusion_matrix(trueList, predList, labels=[1, 2])
		sns.heatmap(C2,annot=True,ax=ax)

		ax.set_title(self.methodName + str(cnt/len(pred)))
		ax.set_xlabel('predict')
		ax.set_ylabel('true')
		plt.savefig(self.output_directory + 'confusion_matrix.png')
		plt.close()
		return cnt/len(pred)
