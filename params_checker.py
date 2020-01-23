import random

from skopt.space import Integer


class params_checker:
    def choice(self, space):
        dVector = {}
        # aVector = []
        # res = []
        for i in space:
            if isinstance(i, Integer):
                dVector[i.name] = random.randint(i.low, i.high)
                #aVector.append(random.choice(space[i]))
            else:
                if i.prior == 'uniform':
                    dVector[i.name] = abs(random.uniform(i.low, i.high))
                elif i.prior == 'log-uniform':
                    dVector[i.name] = abs(random.lognormvariate(i.low, i.high))
                #aVector.append(abs(space[i]))
        #res.append(aVector)
        return dVector