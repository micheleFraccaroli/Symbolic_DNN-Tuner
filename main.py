from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args

from colors import colors
from controller import controller
from dataset.cifar_dataset import cifar_data
from search_space import search_space

X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
log = open("algorithm_logs/hyper-parameters.txt", "a")
controller = controller(log, X_train, Y_train, X_test, Y_test, n_classes)
max_evals = 3

# hyper-parameters
sp = search_space()
search_space = sp.search_sp()

# objective function
@use_named_args(search_space)
def objective(**params):
    f = open("algorithm_logs/hyperparameters.txt", "a")
    to_optimize = controller.training(params)
    f.close()
    return to_optimize

def start_analisys():
    new_space, new_model, to_optimize = controller.diagnosis()

def start(search_space):
    '''
    Starting bayesian Optimization
    :return: research result
    '''
    print(colors.MAGENTA, "--> START BAYESIAN OPTIMIZATION \n", colors.ENDC)

    checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)

    # optimization
    search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                             callback=[checkpoint_saver])
    start_analisys()

    return search_res
    # restore checkpoint
    # search_res = load('checkpoints/checkpoint.pkl')
    # x0 = search_res.x_iters


print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)
search_res = start(search_space)
print(search_res)