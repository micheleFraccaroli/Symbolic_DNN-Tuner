from neural_network import neural_network
from dataset.cifar_dataset import cifar_data
from bayesian_opt import bayesian_opt
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize, callbacks
from skopt import load
from skopt.callbacks import CheckpointSaver

search_space = [
    Integer(16, 48, name='unit_c1'),
    Real(0.002, 0.3, name='dr1_2'),
    Integer(64, 128, name='unit_c2'),
    Integer(256, 512, name='unit_d'),
    Real(0.03, 0.5, name='dr_f'),
    Real(10 ** -5, 10 ** -1, name='learning_rate')
]


# objective function
@use_named_args(search_space)
def objective(**params):
    f = open("hyperparameters.txt", "a")
    n = neural_network(X_train, Y_train, X_test, Y_test, 10, f)
    model, history = n.training(params)
    score = model.evaluate(X_test, Y_test)
    f.close()
    return -score[1]


X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
bo = bayesian_opt(3)
X = bo.choice(search_space)
checkpoint_saver = CheckpointSaver("../checkpoints/checkpoint.pkl", compress=9)

# optimization
search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=10, n_random_starts=1,
                         callback=[checkpoint_saver])

print(search_res)
# restore checkpoints
search_res = load('../checkpoints/checkpoint.pkl')
x0 = search_res.x_iters
