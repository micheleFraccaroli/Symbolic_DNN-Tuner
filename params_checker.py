import random

from skopt.space import Integer, Real


class paramsChecker:
    def checker(self,i, dVector):
        if isinstance(i, Integer):
            dVector[i.name] = random.randint(i.low, i.high)
        else:
            if i.prior == 'uniform':
                dVector[i.name] = abs(random.uniform(i.low, i.high))
            elif i.prior == 'log-uniform':
                dVector[i.name] = abs(random.lognormvariate(i.low, i.high))
        return dVector

    def choice(self, space, toChange=None, params=None):
        dVector = {}
        for i in space:
            if toChange or params:
                if i.name in toChange:
                    dVector = self.checker(i,dVector)
                else:
                    dVector[i.name] = params[i.name]
            else:
                dVector = self.checker(i, dVector)
        return dVector


if __name__ == '__main__':
    search_space = [
        Integer(16, 64, name='unit_c1'),
        Real(0.002, 0.3, name='dr1_2'),
        Integer(64, 128, name='unit_c2'),
        Integer(256, 512, name='unit_d'),
        Real(0.03, 0.5, name='dr_f'),
        Real(10 ** -5, 10 ** -1, name='learning_rate'),
        Integer(16, 128, name='batch_size'),
    ]

    params = {'unit_c1': 64, 'dr1_2': 0.3, 'unit_c2': 127, 'unit_d': 512, 'dr_f': 0.5, 'learning_rate': 1e-05, 'batch_size': 16}
    pc = paramsChecker()
    #dv = pc.choice(search_space, ['learning_rate','batch_size'], params)
    dv = pc.choice(search_space)
    print(dv)
