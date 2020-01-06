# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
import numpy as np
import copy
import pickle
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm, loguniform
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from neural_network import neural_network
from dataset.cifar_dataset import cifar_data
from bayesian_opt import bayesian_opt

# objective function
def objective(f, params):
    # noise = normal(loc=0, scale=noise)
    # return (x ** 2 * sin(5 * pi * x) ** 6.0) + noise
    n = neural_network(X_train, Y_train, X_test, Y_test, 10, f)
    model, history = n.training(params)
    score = model.evaluate(X_test, Y_test)
    return -score[1]

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)


# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]


# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()


X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
bo = bayesian_opt(3)
default_params = {
    'unit_c1': 64,
    'dr1_2': 0.25,
    'unit_c2': 96,
    'unit_d': 512,
    'dr_f': 0.5,
    'learning_rate': 0.002
}
# hyper-parameters
search_space = {
    'unit_c1': [16,32,48,64],
    'dr1_2': np.random.normal(0.002, 0.3),
    'unit_c2': [64,80,96,112,128],
    'unit_d': [256,512],
    'dr_f': np.random.normal(0.3,0.5),
    'learning_rate': loguniform.rvs(10**-5, 10**-1)
}

f = open("hyperparameters.txt", "w")

X, Xx = bo.choice(search_space)
y = objective(f, X)

XX = copy.copy(Xx)
xxx = np.array(XX).reshape(-1,1)
#xxx = xxx.reshape(xxx.size,1)

yy = np.array(copy.copy(y))
yy = np.repeat(yy,6)
yy = yy.reshape(-1,1)

print("------------------------------------------------------------")
f.close()

# reshape into rows and cols
# X = X.reshape(len(X), 1)
# y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(xxx, yy)
# plot before hand
plot(xxx, yy, model)
# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
