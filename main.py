import NetworkBuilder
import os as os

net = NetworkBuilder.NetworkBuilder()

X, y = net.loadData()

model = net.generateModel()

fit_model = net.fitModel(X,y,model)

yhat = net.makePredictions(fit_model, X)




