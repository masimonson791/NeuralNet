import NetworkBuilder
import os as os

net = NetworkBuilder.NetworkBuilder()

X, y = net.loadData()
y0 = y[:,0]
y1 = y[:,1]

mlp_model = net.generateMLP()

mlp_fit_model = net.fitModel(X,[y0,y1],mlp_model)

mlp_model = net.loadWeights(model, "0001.hdf5")

# examine model:
mlp_model.summary()

mlp_yhat = net.makePredictions(model, X)

#---------------------------------

lstm_model = net.generateLSTM()

# reshape for LSTM format
X = X.reshape(X.shape[0],1,X.shape[1])

lstm_fit_model = net.fitModel(X,[y0,y1],lstm_model)

lstm_model = net.loadWeights(lstm_model, "0001.hdf5")

lstm_yhat = net.makePredictions(lstm_model, X)

#---------------------------------

cnn_model = net.generateCNN1D()

# reshape for CNN format
X = X.reshape(X.shape[0],X.shape[1],1)

cnn_fit_model = net.fitModel(X,[y0,y1],cnn_model)

cnn_model = net.loadWeights(cnn_model, "0001.hdf5")

cnn_yhat = net.makePredictions(cnn_model, X)