import datetime
import time

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from tensorflow.keras import backend as K

from colors import colors
from controller import controller
from datasets.cifar_dataset import cifar_data
from search_space import search_space
from params_checker import paramsChecker

# MNIST SECTION --------------------------------------------------------------------------------------------------------

# X_train, X_test, Y_train, Y_test, n_classes = mnist()
# CIFAR-10 SECTION -----------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
dt = datetime.datetime.now()
max_evals = 20

# hyper-parameters
sp = search_space()
spa = sp.search_sp()
controller = controller(X_train, Y_train, X_test, Y_test, n_classes)

# objective function
space = {}
start_time = time.time()


def update_space(new_space):
    global search_space
    search_space = new_space
    return search_space


search_space = update_space(spa)


def objective(params):
    space = {}
    for i, j in zip(search_space, params):
        space[i.name] = j
    print(space)
    f = open("algorithm_logs/hyper-neural.txt", "a")
    f.write(str(space) + "\n")
    to_optimize = controller.training(space)
    f.close()
    K.clear_session()
    return to_optimize


def start_analisys():
    new_space, to_optimize = controller.diagnosis()
    return new_space, to_optimize


def check_continuing_BO(new_space, x_iters, func_vals):
    func_vals = func_vals.tolist()
    for x in x_iters:
        for n, i in zip(new_space, x):
            if i < n.low or i > n.high:
                _ = func_vals.pop(x_iters.index(x))
                x_iters.remove(x)
                break
    return x_iters, func_vals


def start(search_space, iter):
    """
    Starting bayesian Optimization
    :return: research result
    """
    print(colors.MAGENTA, "|  ----------- START BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

    checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)
    # optimization
    controller.set_case(False)
    search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                             callback=[checkpoint_saver])

    # K.clear_session()
    new_space, to_optimize = start_analisys()

    for opt in range(iter):
        # restore checkpoint
        if len(new_space) == len(search_space):
            # controller.set_case(True)
            res = load('checkpoints/checkpoint.pkl')
            try:
                print(new_space)
                search_res = gp_minimize(objective, new_space, x0=res.x_iters, y0=res.func_vals, acq_func='EI',
                                         n_calls=1,
                                         n_random_starts=0, callback=[checkpoint_saver])
                print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
                print(colors.FAIL, "Inside BO", colors.ENDC)
                print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
            except:
                print(new_space)
                #res.x_iters, res.func_vals = check_continuing_BO(new_space, res.x_iters, res.func_vals)
                search_res = gp_minimize(objective, new_space, y0=res.func_vals, acq_func='EI',
                                         n_calls=1,
                                         n_random_starts=1, callback=[checkpoint_saver])
                print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
                print(colors.WARNING, "Other BO", colors.ENDC)
                print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
        else:
            search_space = update_space(new_space)
            search_res = gp_minimize(objective, new_space, acq_func='EI', n_calls=1, n_random_starts=1,
                                     callback=[checkpoint_saver])

        new_space, to_optimize = start_analisys()

    return search_res


print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)
search_res = start(search_space, max_evals)
print(search_res)
print(colors.OKGREEN, "\nEND ALGORITHM \n", colors.ENDC)
end_time = time.time()

print(colors.CYAN, "\nTIME --------> \n", end_time - start_time, colors.ENDC)
