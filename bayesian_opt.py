import numpy as np
from scipy.stats import norm
from warnings import simplefilter, catch_warnings
import random


class bayesian_opt:
    def __init__(self, max_evals):
        self.max_evals = max_evals

    def choice(self, space):
        dVector = {}
        aVector = []
        res = []
        for i in space:
            if isinstance(space[i], list):
                dVector[i] = random.choice(space[i])
                aVector.append(random.choice(space[i]))
            else:
                dVector[i] = abs(space[i])
                aVector.append(abs(space[i]))
        res.append(aVector)
        return dVector, res

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
        best = max(mfar)

        # calculate mean and std dev via surrogate
        mu, std = self.surrogate(model, new_X)
        mu = mu[:, 0]

        # probability of improvement
        PoI = norm.cdf((mu - best) / (std + 1E-9))
        return PoI

    def opt_aquisition(self, model, space):
        '''
        - create selection of new hyper-parameters -> random selection from dictionary or smart
          selection from other function (use random for initial test)
        - call self.acquisition for retrieve score probability
        - return best hp for the best score

         https://machinelearningmastery.com/what-is-bayesian-optimization/
        '''
        vector = self.choice(space)
        vector = vector.reshape(len(vector),1)

        # calculate the acquisition function
        scores = self.acquisition(model, space, vector)
        max = np.argmax(scores)
        return vector[max,0]


    # def byOptimize(self, space, init):
    #     '''
    #     Perform Bayesian Optimization
    #     :param space: hyper-parameters space
    #     :param init: first hyper-parameters for first training
    #     :return: best hyper-parameters founded
    #     '''
    #     model = GaussianProcessRegressor()
    #     model.fit(space,init)
    #
    #     for i in range(self.max_evals):
    #         x = self.opt_aquisition(model, space)
    #         actual = self.obj(x)
