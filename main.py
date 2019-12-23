import numpy as np
from neural_network import net as cnet
from dataset.cifar_dataset import cifar_data
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from tensorflow.keras import backend as K
from colors import colors


'''
hp → hyperparameter space
fmin → minimizing the function
tpe → Tree-structured Parzen Estimator model (optimization algorithm)
'''
f = open("hyper-parameters.txt", "w")

# creating optimization algorithm & create trials object
tpe_alg = tpe.suggest
tpe_trial = Trials()

# datasets
X_train, X_test, Y_train, Y_test, n_classes = cifar_data()


def testing(model, data, labels):
    score = model.evaluate(data, labels)
    return score


default_params = {
    'unit_c1': 32,
    'dr1_2': 0.25,
    'unit_c2': 64,
    'unit_d': 512,
    'dr_f': 0.5,
    'learning_rate': 0.002
}

model = cnet(X_train, Y_train, X_test, Y_test, default_params, f)
print("FIRST LOSS, FIRST ACCURACY: ", testing(model, X_test, Y_test))

# hyper-parameters
search_space = {
    'unit_c1': hp.choice('unit_c1', np.arange(16, 65, 16)),
    'dr1_2': hp.uniform('dr1_2', 0.002, 0.3),
    'unit_c2': hp.choice('unit_c2', np.arange(64, 129, 16)),
    'unit_d': hp.choice('unit_d', np.arange(256, 1025, 256)),
    'dr_f': hp.uniform('dr_f', 0.3, 0.5),
    'learning_rate': hp.loguniform('learning_rate', -10, 0),
}


# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, stratify=Y_train)

def hyperopt_fcn(params):
    model = cnet(X_train, Y_train, X_test, Y_test, params, f)
    test = testing(model, X_test, Y_test)
    K.clear_session()
    return {'loss': -test[1], 'status': STATUS_OK}


# optimization
print(colors.WARNING, "\n------------ START BAYESIAN OPTIMIZATION ------------\n", colors.ENDC)
print(colors.WARNING, "OPTIMIZING ...", colors.ENDC)
best = fmin(hyperopt_fcn, search_space, algo=tpe_alg, max_evals=10)

print("=============================== BEST ===============================\n")
print(best)
print("====================================================================\n")

f.close()
print("\n\n------------------------ SPACE EVALS RESULTS -------------------------- \n")
print(space_eval(search_space, best))
print("---------------------------------------------------------------------------")
