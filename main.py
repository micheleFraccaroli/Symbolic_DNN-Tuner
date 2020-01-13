import datetime
import sys

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations

from colors import colors
from controller import controller
from dataset.cifar_dataset import cifar_data
from params_checker import params_checker
from search_space import search_space
from tensorflow.keras import backend as K
from objFunction import objFunction

X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
dt = datetime.datetime.now()
max_evals = 3

# hyper-parameters
sp = search_space()
search_space = sp.search_sp()
pc = params_checker()
first_params = pc.choice(search_space)
controller = controller(X_train, Y_train, X_test, Y_test, n_classes, first_params)

# objective function
space = {}
def objective(params):
    for i, j in zip(search_space, params):
        space[i.name] = j

    f = open("algorithm_logs/hyper-neural.txt", "a")
    f.write(str(space) + "\n")
    to_optimize = controller.training(space)
    f.close()
    return to_optimize

def start_analisys():
    new_space, new_model, to_optimize = controller.diagnosis()
    return new_space, new_model, to_optimize

def start(search_space, iter):
    '''
    Starting bayesian Optimization
    :return: research result
    '''
    print(colors.MAGENTA, "|  ----------- START BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

    checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)
    # optimization
    controller.set_case(True)
    search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1, callback=[checkpoint_saver])
    new_space, new_model, to_optimize = start_analisys()

    for opt in range(iter):
        # restore checkpoint
        if len(new_space) == len(search_space):
            controller.set_case(False)
            res = load('checkpoints/checkpoint.pkl')
            search_res = gp_minimize(objective, new_space, y0=res.func_vals, acq_func='EI', n_calls=10,
                                     n_random_starts=1, callback=[checkpoint_saver])
        else:
            controller.set_case(True)
            search_space = new_space
            search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=10, n_random_starts=1,
                                     callback=[checkpoint_saver])
        new_space, new_model, to_optimize = start_analisys()

    return search_res


print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)
search_res = start(search_space, max_evals)
plot_convergence(search_res)
print(search_res)