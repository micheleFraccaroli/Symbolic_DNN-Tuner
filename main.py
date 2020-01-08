from colors import colors
from controller import controller
from neural_network import neural_network
from dataset.cifar_dataset import cifar_data
from bayesian_opt import bayesian_opt
from skopt.utils import use_named_args
from skopt import gp_minimize, callbacks
from skopt import load
from skopt.callbacks import CheckpointSaver
from search_space import search_space

class main_class:
    def __init__(self, search_space, X_train, Y_train, X_test, Y_test, n_classes):
        #self.space = search_space
        self.log = open("hyper-parameters.txt", "a")
        self.controller = controller(self.log, X_train, Y_train, X_test, Y_test, n_classes)
        self.max_evals = 3

    # objective function
    @use_named_args(dimensions=search_space.search_sp())
    def objective(self,**params):
        f = open("hyperparameters.txt", "a")
        # n = neural_network(X_train, Y_train, X_test, Y_test, 10, f)
        # model, history = n.training(params)
        # score = model.evaluate(X_test, Y_test)
        to_optimize = self.controller.training(params)
        f.close()
        return to_optimize

    # def start(self, space, init=False):
    #     '''
    #     Starting training for Bayesian Optimization
    #     :return: params to optimize
    #     '''
    #     to_optimize = self.controller.training(space,init)
    #     return to_optimize

    def start(self):
        '''
        Starting bayesian Optimization
        :return:
        '''
        print(colors.WARNING, "\n------------ START BAYESIAN OPTIMIZATION ------------\n", colors.ENDC)
        print(colors.WARNING, "OPTIMIZING ...", colors.ENDC)

        checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)

        # optimization
        search_res = gp_minimize(self.objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                                 callback=[checkpoint_saver])


        # restore checkpoint
        search_res = load('checkpoints/checkpoint.pkl')
        x0 = search_res.x_iters


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, n_classes = cifar_data()
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, stratify=Y_train)

    # hyper-parameters
    sp = search_space()
    search_space = sp.search_sp()
    # search_space = [
    #     Integer(16, 48, name='unit_c1'),
    #     Real(0.002, 0.3, name='dr1_2'),
    #     Integer(64, 128, name='unit_c2'),
    #     Integer(256, 512, name='unit_d'),
    #     Real(0.03, 0.5, name='dr_f'),
    #     Real(10 ** -5, 10 ** -1, name='learning_rate')
    # ]
    launch = main_class(search_space, X_train, Y_train, X_test, Y_test, n_classes)
    print(colors.OKGREEN, "\n------------ START ALGORITHM ------------\n", colors.ENDC)
    launch.start()