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

class Classifier_CNNMLP:

	def __init__(self, output_directory, inputA_shape, inputB_shape, nb_classes, verbose=False,build=True,flag_att=0,flag_CBAM=0):
		self.output_directory = output_directory
		self.flag_CBAM = flag_CBAM
		self.flag_att = flag_att
		if build == True:
			self.model = self.build_model(inputA_shape, inputB_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			# self.model.save_weights(self.output_directory+'model_init.hdf5')
			self.methodName = "cnnMLP"
		return

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
		cbam_feature = self.channel_attention(cbam_feature, ratio)
		cbam_feature = self.spatial_attention(cbam_feature, )
		return cbam_feature

	def build_model(self, inputA_shape, inputB_shape, nb_classes, n_feature_maps=64):
		inputA_layer = keras.layers.Input(inputA_shape)
		inputB_layer = keras.layers.Input(inputB_shape)

		# A part, cgm
		conv1A = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(inputA_layer)
		conv1A = keras.layers.BatchNormalization()(conv1A)
		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv1A)
			conv1A = keras.layers.add([conv1A, cbam])
			conv1A = keras.layers.Activation(activation='relu')(conv1A)

		conv2A = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1A)
		conv2A = keras.layers.BatchNormalization()(conv2A)
		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv2A)
			conv2A = keras.layers.add([conv2A, cbam])
			conv2A = keras.layers.Activation('relu')(conv2A)

		conv3A = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2A)
		conv3A = keras.layers.BatchNormalization()(conv3A)
		if self.flag_CBAM == 1:
			cbam = self.cbam_block(conv3A)
			conv3A = keras.layers.add([conv3A, cbam])
			conv3A = keras.layers.Activation('relu')(conv3A)

		pooling_layerA = keras.layers.GlobalAveragePooling1D()(conv3A)
		dense_layerA = keras.layers.Dense(64, activation='relu')(pooling_layerA)
		
		modelA = keras.models.Model(inputs=inputA_layer, outputs=dense_layerA)

		# B part, auxiliaries
		input_layer_flattened = keras.layers.Flatten()(inputB_layer)

		layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
		layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

		layer_2 = keras.layers.Dropout(0.2)(layer_1)
		layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

		output_layerB = keras.layers.Dropout(0.3)(layer_2)
		output_layerB = keras.layers.Dense(64, activation='relu')(output_layerB)

		modelB = keras.models.Model(inputs=inputB_layer, outputs=output_layerB)

		# combine
		combined = concatenate([modelA.output, modelB.output])
		combined = tf.expand_dims(combined, -1)

		# attention
		if self.flag_att == 1:
			weight = keras.layers.Dense(128, activation='softmax')(combined)
			combined = weight * combined
		
		conv1 = keras.layers.Conv1D(16, kernel_size=2, padding='same')(combined)
		conv1 = keras.layers.BatchNormalization()(conv1)
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

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_valAB)
		np.save(self.output_directory + 'fcnMLP_y_pred.npy', y_pred)
		np.save(self.output_directory + 'fcnMLP_y_true.npy', y_true)
		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_testAB, y_true, x_trainAB, y_train, y_test, return_df_metrics = False):
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
