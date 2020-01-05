import NetworkBuilder
import os as os

net = NetworkBuilder.NetworkBuilder()

X, y = net.loadData()
y0 = y[:,0]
y1 = y[:,1]

model = net.generateModel()

fit_model = net.fitModel(X,[y0,y1],model)

model = net.loadWeights(model, "0001.hdf5")

yhat = net.makePredictions(model, X)




