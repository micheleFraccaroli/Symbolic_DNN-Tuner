from neural_network import neural_network
from tensorflow.keras import backend as K
from dataset.cifar_dataset import cifar_data
import sys
import time

from hyperopt import space_eval
from hyperopt import fmin, tpe, hp, Trials
from hyperopt import STATUS_OK

X_train, X_test, Y_train, Y_test, n_classes = cifar_data()

default_params = {
    'unit_c1': hp.uniform('unit_c1', 16, 64),
    'dr1_2': hp.uniform('dr1_2', 0.002, 0.3),
    'unit_c2': hp.uniform('unit_c2', 64, 128),
    'unit_d': hp.uniform('unit_d', 256, 512),
    'dr_f': hp.uniform('dr_f', 0.03, 0.5),
    'learning_rate': hp.uniform('learning_rate', 10 ** -5, 10 ** -1),
    'batch_size': hp.uniform('batch_size', 16, 128),
}
start_time = time.time()

def objective(params):
    params['unit_c1'] = int(params['unit_c1'])
    params['unit_c2'] = int(params['unit_c2'])
    params['unit_d'] = int(params['unit_d'])
    params['batch_size'] = int(params['batch_size'])
    f = open("hyperparameters.txt", "a")
    f.write(str(params))
    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    score, history, model = n.training(params, False)
    f.close()
    K.clear_session()
    return {'loss': -score[1], 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, default_params, algo=tpe.suggest, max_evals=20, trials=trials)
end_time = time.time()

print("TIME: ")
print(end_time - start_time)