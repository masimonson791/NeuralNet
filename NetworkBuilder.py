from numpy import loadtxt
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.regularizers import l1
from keras.regularizers import l2
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv1D,UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import numpy as np
import os as os
import configparser as cp
import time as time

# split into input (X) and output (y) variables

class NetworkBuilder:

	def loadData(self):		
		dataset = loadtxt(self.DATA, delimiter=",")
		X = dataset[:,self.X_COLUMNS]
		y = dataset[:,self.Y_COLUMNS]
		def min_max(x):
			return [np.round((xx-np.min(x))/(1.0*(np.max(x)-np.min(x))),2) for xx in x]
		for i in range(X.shape[1]):
			X[:,i] = min_max(X[:,i])
		for i in range(y.shape[1]):
			y[:,i] = min_max(y[:,i])
		return X, y

	def generateMLP(self):
		layer_0 = Input(shape=(len(self.X_COLUMNS),))
		ltype = self.REGTYPE
		if self.REGTYPE == "none":
			ACREG=""
		if self.REGTYPE != "none":	
			ACREG = "activity_regularizer="+ltype+"("+self.REGVAL+")"
		for i in range(self.NUM_HIDDEN_LAYERS):
			i = i*2
			i = i+1
			layer_i="layer_"+str(i)+" = Dense("+str(self.HL_UNITS)+", activation='relu',"+ACREG+",)(layer_"+(str(i-1))+")\nlayer_"+str(i+1)+" = layers.Dropout("+self.DROPVAL+")(layer_"+(str(i))+")"
			exec(layer_i)	
		layer_out = []
		layer_out.clear()
		for i in range(len(self.LOSS)):
			if self.LOSS[i]=='mean_squared_error':
				actif = 'linear'
			if self.LOSS[i]=='categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_absolute_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='binary_crossentropy':
				actif = 'sigmoid'
			if self.LOSS[i]=='hinge':
				actif = 'tanh'
			if self.LOSS[i]=='squared_hinge':
				actif = 'tanh'
			if self.LOSS[i]=='sparse_categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='kullback_libler_divergence':
				actif = 'softmax'							
			layer_out_i="layer_out_"+str(i)+" = Dense(1, activation='"+actif+"')(layer_"+str(self.NUM_HIDDEN_LAYERS*2)+")"
			exec(layer_out_i)
			layer_out.append("layer_out_"+str(i)+"")
		outputs = ",".join(layer_out)
		model = eval("Model(inputs=layer_0, outputs=["+outputs+"])")
		# compile the keras model
		opt = SGD(lr=self.LR, momentum=self.MOMENTUM, decay=self.DECAY)
		model.compile(loss=self.LOSS, optimizer=opt, metrics=self.METRICS)
		print(model.summary())
		return(model)

	def generateLSTM(self):
		layer_0 = Input(shape=(1,len(self.X_COLUMNS),))
		ltype = self.REGTYPE
		if self.REGTYPE == "none":
			ACREG=""
		if self.REGTYPE != "none":	
			ACREG = "activity_regularizer="+ltype+"("+self.REGVAL+")"
		for i in range(self.NUM_HIDDEN_LAYERS):
			if(i==(self.NUM_HIDDEN_LAYERS-1)):
				i = i*2
				i = i+1
				layer_i="layer_"+str(i)+" = LSTM("+str(self.HL_UNITS)+", activation='relu',"+ACREG+",return_sequences=False)(layer_"+(str(i-1))+")\nlayer_"+str(i+1)+" = layers.Dropout("+self.DROPVAL+")(layer_"+(str(i))+")"
				print("last layer")
			else:
				print(i)
				i = i*2
				i = i+1
				layer_i="layer_"+str(i)+" = LSTM("+str(self.HL_UNITS)+", activation='relu',"+ACREG+",return_sequences=True)(layer_"+(str(i-1))+")\nlayer_"+str(i+1)+" = layers.Dropout("+self.DROPVAL+")(layer_"+(str(i))+")"		
			exec(layer_i)	
		layer_out = []
		layer_out.clear()
		for i in range(len(self.LOSS)):
			if self.LOSS[i]=='mean_squared_error':
				actif = 'linear'
			if self.LOSS[i]=='categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_absolute_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='binary_crossentropy':
				actif = 'sigmoid'
			if self.LOSS[i]=='hinge':
				actif = 'tanh'
			if self.LOSS[i]=='squared_hinge':
				actif = 'tanh'
			if self.LOSS[i]=='sparse_categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='kullback_libler_divergence':
				actif = 'softmax'							
			layer_out_i="layer_out_"+str(i)+" = Dense(1, activation='"+actif+"')(layer_"+str(self.NUM_HIDDEN_LAYERS*2)+")"
			exec(layer_out_i)
			layer_out.append("layer_out_"+str(i)+"")
		outputs = ",".join(layer_out)
		model = eval("Model(inputs=layer_0, outputs=["+outputs+"])")
		# compile the keras model
		opt = SGD(lr=self.LR, momentum=self.MOMENTUM, decay=self.DECAY)
		model.compile(loss=self.LOSS, optimizer=opt, metrics=self.METRICS)
		print(model.summary())
		return(model)

	def generateCNN1D(self):
		layer_0 = Input(shape=(len(self.X_COLUMNS),1,))
		ltype = self.REGTYPE
		if self.REGTYPE == "none":
			ACREG=""
		if self.REGTYPE != "none":	
			ACREG = "activity_regularizer="+ltype+"("+self.REGVAL+")"
		for i in range(self.NUM_HIDDEN_LAYERS):
			if(i==(self.NUM_HIDDEN_LAYERS-1)):
				i = i*2
				i = i+1
				layer_i="layer_"+str(i)+" = Conv1D("+str(self.HL_UNITS)+",kernel_size=2, activation='relu',"+ACREG+")(layer_"+(str(i-1))+")\nlayer_"+str(i+1)+" = layers.Flatten()(layer_"+(str(i))+")"
				print("last layer")
			else:
				print(i)
				i = i*2
				i = i+1
				layer_i="layer_"+str(i)+" = Conv1D("+str(self.HL_UNITS)+",kernel_size=2, activation='relu',"+ACREG+")(layer_"+(str(i-1))+")\nlayer_"+str(i+1)+" = layers.Dropout("+self.DROPVAL+")(layer_"+(str(i))+")"		
			exec(layer_i)
		layer_out = []
		layer_out.clear()
		for i in range(len(self.LOSS)):
			if self.LOSS[i]=='mean_squared_error':
				actif = 'linear'
			if self.LOSS[i]=='categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_absolute_error':
				actif = 'linear'
			if self.LOSS[i]=='mean_squared_logarithmic_error':
				actif = 'linear'
			if self.LOSS[i]=='binary_crossentropy':
				actif = 'sigmoid'
			if self.LOSS[i]=='hinge':
				actif = 'tanh'
			if self.LOSS[i]=='squared_hinge':
				actif = 'tanh'
			if self.LOSS[i]=='sparse_categorical_crossentropy':
				actif = 'softmax'
			if self.LOSS[i]=='kullback_libler_divergence':
				actif = 'softmax'							
			layer_out_i="layer_out_"+str(i)+" = Dense(1, activation='"+actif+"')(layer_"+str(self.NUM_HIDDEN_LAYERS*2)+")"
			exec(layer_out_i)
			layer_out.append("layer_out_"+str(i)+"")
		outputs = ",".join(layer_out)
		model = eval("Model(inputs=layer_0, outputs=["+outputs+"])")
		# compile the keras model
		opt = SGD(lr=self.LR, momentum=self.MOMENTUM, decay=self.DECAY)
		model.compile(loss=self.LOSS, optimizer=opt, metrics=self.METRICS)
		print(model.summary())
		return(model)

	def loadWeights(self,model,fileName):
		model.load_weights(fileName)
		return(model)

	def fitModel(self,X,y,model):
		os.system("rm -r my_log_dir")
		os.mkdir("my_log_dir")
		tensorboard = TensorBoard(log_dir="my_log_dir")
		checkpoint = ModelCheckpoint(self.ID+".hdf5", verbose=1, save_best_only=False, mode='max')
		callbacks_list = [checkpoint,tensorboard]
		history = model.fit(X, y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,callbacks=callbacks_list)
		print("enter in terminal to use TensorBoard 2.0.2 at http://localhost:6006/ : 'tensorboard --logdir=my_log_dir --port=6006'")
		return(model)

	def makePredictions(self,model,X):
		yhat = model.predict(X)
		return(yhat)   

# code variational auto-encoder:

# Input sequence dimensions
steps, original_dim = 1, 28*28 # Take care here since we are changing this according to the data
# Number of convolutional filters to use
filters = 64
# Convolution kernel size
num_conv = 6
# Set batch size
batch_size = 100
# Decoder output dimensionality
decOutput = 10

latent_dim = 20
intermediate_dim = 256
epsilon_std = 1.0
epochs = 5

x = Input(batch_shape=(batch_size,steps,original_dim))
# Play around with padding here, not sure what to go with.
conv_1 = Conv1D(1,
                kernel_size=num_conv,
                padding='same', 
                activation='relu')(x)
conv_2 = Conv1D(filters,
                kernel_size=num_conv,
                padding='same', 
                activation='relu',
                strides=1)(conv_1)
flat = Flatten()(conv_2) # Since we are passing flat data anyway, we probably don't need this.
hidden = Dense(intermediate_dim, activation='relu')(flat)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var ) * epsilon # the original VAE divides z_log_var with two -- why?

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])



# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # Double check wtf this is supposed to be
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='adam', loss=vae_loss) # 'rmsprop'
vae.summary()

	def __init__(self):

		def ConfigSectionMap(section):
			dict1 = {}
			options = Config.options(section)
			for option in options:
			    try:
			        dict1[option] = Config.get(section, option)
			        if dict1[option] == -1:
			            DebugPrint("skip: %s" % option)
			    except:
			        print("exception on %s!" % option)
			        dict1[option] = None
			return dict1	
		self.ROOT_PATH = '/Volumes/HD/Code/python/2019/DeepAdvisors/NeuralNet/'
		Config = cp.ConfigParser()
		Config.read(self.ROOT_PATH+"config.ini")
		sections = Config.sections()[0]
		self.X_COLUMNS = list(ConfigSectionMap(sections)['x_columns'].split(','))
		for i in range(len(self.X_COLUMNS)):
			self.X_COLUMNS[i] = int(self.X_COLUMNS[i])
		self.Y_COLUMNS = list(ConfigSectionMap(sections)['y_columns'].split(','))
		for i in range(len(self.Y_COLUMNS)):
			self.Y_COLUMNS[i] = int(self.Y_COLUMNS[i])
		self.LOSS = ConfigSectionMap("NetworkParameters")['loss'].split(',')
		self.METRICS = ConfigSectionMap("NetworkParameters")['metrics'].split(',')
		self.NUM_HIDDEN_LAYERS = int(ConfigSectionMap("NetworkParameters")['num_hidden_layers'])
		self.HL_UNITS = int(ConfigSectionMap("NetworkParameters")['hl_units'])
		self.EPOCHS = int(ConfigSectionMap("NetworkParameters")['epochs'])
		self.BATCH_SIZE = int(ConfigSectionMap("NetworkParameters")['batch_size'])
		self.DATA = ConfigSectionMap("NetworkParameters")['data']
		self.LR = float(ConfigSectionMap("NetworkParameters")['lr'])
		self.MOMENTUM = float(ConfigSectionMap("NetworkParameters")['momentum'])
		self.DECAY = float(ConfigSectionMap("NetworkParameters")['decay'])
		self.REGTYPE = ConfigSectionMap("NetworkParameters")['regularization_type']
		self.REGVAL = ConfigSectionMap("NetworkParameters")['regularization_val']
		self.ID = ConfigSectionMap("NetworkParameters")['unique_model_id']
		self.DROPVAL = ConfigSectionMap("NetworkParameters")['dropout_val']
