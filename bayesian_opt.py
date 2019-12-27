import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from warnings import simplefilter, catch_warnings

class bayesian_opt:
    def __init__(self, obj_function, max_evals):
        self.obj = obj_function
        self.max_evals = max_evals

    def surrogate(self, model, X):
        '''
        Function for creatre surrogate surface
        :param model: model
        :param X: data
        :return: mean and std_deviation of the model on data
        '''

        with catch_warnings():
            simplefilter('ignore')
            return model.predict(X, return_std=True)

    def acquisition(self, model, space, new_X):
        '''
        Acquisition function
        :param model: model of obj_funct
        :param space: hp space
        :param new_X: new choice of hp space
        :return: PoI (Probability of Improvement)
        '''
        # best surrogate score found so far
        mfar, _ = self.surrogate(model, space)
        best = np.argmax(mfar)

        # calculate mean and std dev via surrogate
        mu, std = self.surrogate(model, new_X)
        mu = mu[:,0]

        # probability of improvement
        PoI = norm.cdf((mu-best) / (std+1E-9))
        return PoI

    def opt_aquisition(self, model, space, y):
        '''
        - create selection of new hyper-parameters -> random selection from dictionary or smart
          selection from other function (use random for initial test
        - call self.acquisition for retrieve score probability
        - return best hp for the best score

         https://machinelearningmastery.com/what-is-bayesian-optimization/
        '''

    def optimize(self):
        model = GaussianProcessRegressor()
