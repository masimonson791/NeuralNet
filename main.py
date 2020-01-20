import NetworkBuilder
import os as os

net = NetworkBuilder.NetworkBuilder()

X, y = net.loadData()
y0 = y[:,0]
y1 = y[:,1]

model = net.generateMLP()

fit_model = net.fitModel(X,[y0,y1],model)

model = net.loadWeights(model, "0001.hdf5")

# examine model:
model.summary()

yhat = net.makePredictions(model, X)

lmtm_model = net.generateLSTM()

lstm_fit_model = net.fitModel(X,[y0,y1],lstm_model)

lstm_model = net.loadWeights(lstm_model, "0001.hdf5")

lstm_yhat = net.makePredictions(lstm_model, X)