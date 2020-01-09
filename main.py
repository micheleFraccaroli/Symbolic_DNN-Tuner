import datetime
import sys

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args

from colors import colors
from controller import controller
from dataset.cifar_dataset import cifar_data
from params_checker import params_checker
from search_space import search_space
from tensorflow.keras import backend as K
from objFunction import objFunction

X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
dt = datetime.datetime.now()
log = open("algorithm_logs/hyper-parameters.txt", "a")
log.write("-------------- " + str(dt) + " --------------\n")
max_evals = 3

# hyper-parameters
sp = search_space()
search_space = sp.search_sp()
pc = params_checker()
first_params = pc.choice(search_space)
controller = controller(log, X_train, Y_train, X_test, Y_test, n_classes, first_params)

# objective function
space = {}
def objective(params):
    print(search_space)
    for i, j in zip(search_space, params):
        space[i.name] = j

    f = open("algorithm_logs/hyper-neural.txt", "a")
    f.write(str(space))
    to_optimize = controller.training(space)
    f.close()
    return to_optimize
# @use_named_args(search_space)
# def objective(**params):
#     f = open("algorithm_logs/hyperparameters.txt", "a")
#     print(params)
#     to_optimize = controller.training(params)
#     f.close()
#     return to_optimize


def start_analisys():
    new_space, new_model, to_optimize = controller.diagnosis()
    return new_space, new_model, to_optimize


def start(search_space, iter):
    '''
    Starting bayesian Optimization
    :return: research result
    '''
    print(colors.MAGENTA, "--> START BAYESIAN OPTIMIZATION \n", colors.ENDC)

    checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)
    # optimization
    search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                             callback=[checkpoint_saver])
    new_space, new_model, to_optimize = start_analisys()

    for opt in range(iter):
        # restore checkpoint
        if len(new_space) == len(search_space):
            res = load('checkpoints/checkpoint.pkl')
            #K.clear_session()
            search_res = gp_minimize(objective, new_space, x0=res.x_iters, y0=res.func_vals, acq_func='EI', n_calls=1,
                                     n_random_starts=1, callback=[checkpoint_saver])
        else:
            search_space = new_space
            search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                                     callback=[checkpoint_saver])
        new_space, new_model, to_optimize = start_analisys()

    return search_res


print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)
search_res = start(search_space, max_evals)
print(search_res)