from skopt.space import Integer, Real


class search_space:
    def __init__(self):
        self.epsilon_r1 = 10**-3
        self.epsilon_r2 = 10**2
        self.epsilon_i = 2
        self.epsilon_d = 4

    def search_sp(self):
        self.search_space = [
            Integer(16, 64, name='unit_c1'),
            Real(0.002, 0.3, name='dr1_2'),
            Integer(64, 128, name='unit_c2'),
            Integer(256, 512, name='unit_d'),
            Real(0.03, 0.5, name='dr_f'),
            Real(10 ** -5, 10 ** -1, name='learning_rate'),
            Integer(16, 256, name='batch_size'),
        ]

        return self.search_space

    def add_params(self, params):
        new_Hp = []
        for p in params.keys():
            if type(params[p]) == float:
                np = Real(abs(params[p]/self.epsilon_r2), (params[p]/self.epsilon_r1), name=p)
                new_Hp.append(np)
            elif type(params[p]) == int:
                if p == 'new_fc':
                    np = Integer(abs(int(params[p] / self.epsilon_d)), params[p] * self.epsilon_i, name=p)
                else:
                    np = Integer(abs(params[p] - self.epsilon_i), params[p] + self.epsilon_i, name=p)
                new_Hp.append(np)
        return self.search_space + new_Hp


if __name__ == '__main__':
    ss = search_space()
    sp = ss.search_sp()

    dtest = {'reg': 1e-4}
    res_final = ss.add_params(dtest)
    print(sp)
    print("-----------------------")
    print(res_final)
