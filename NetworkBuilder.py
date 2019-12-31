from numpy import loadtxt
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
import numpy as np
import os as os
import configparser as cp

# split into input (X) and output (y) variables

# define model parameters:
#MODEL_TYPE = "Regression" # "Regression", "Binary Classification", "Multi-Class Classification"


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

	def generateModel(self):
		layer_0 = Input(shape=(len(self.X_COLUMNS),))
		for i in range(self.NUM_HIDDEN_LAYERS):
			i = i+1
			layer_i="layer_"+str(i)+" = Dense("+str(self.HL_UNITS)+", activation='relu')(layer_"+(str(i-1))+")"
			exec(layer_i)	
		layer_out = []
		layer_out.clear()
		for i in range(len(self.LOSS)):
			if self.LOSS[i]=='mean_squared_error':
				actif = 'relu'
			if self.LOSS[i]=='categorical_crossentropy':
				actif = 'softmax'
			layer_out_i="layer_out_"+str(i)+" = Dense(1, activation='"+actif+"')(layer_"+str(self.NUM_HIDDEN_LAYERS)+")"
			exec(layer_out_i)
			layer_out.append("layer_out_"+str(i)+"")
		outputs = ",".join(layer_out)	
		exec("model = Model(inputs=layer_0, outputs=["+outputs+"])")
		print(model.summary())
		# compile the keras model
		opt = SGD(lr=self.LR, momentum=self.MOMENTUM, decay=self.DECAY)
		model.compile(loss=self.LOSS, optimizer=opt, metrics=self.METRICS)
		return(model)

	def fitModel(self,X,y,model):
		model.fit(X, y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)
		return(model)

	def makePredictions(self,model,X):
		yhat = model.predict(X)
		return(yhat)    

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
		self.ROOT_PATH = '/Volumes/HD/Code/python/2019/DeepAdvisors/'
		Config = cp.ConfigParser()
		Config.read(self.ROOT_PATH+"/HO_MLP/mlp/config.ini")
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
