from skopt.space import Integer, Real

class search_space:
    def search_sp(self):
        self.search_space = [
            Integer(16, 48, name='unit_c1'),
            Real(0.002, 0.3, name='dr1_2'),
            Integer(64, 128, name='unit_c2'),
            Integer(256, 512, name='unit_d'),
            Real(0.03, 0.5, name='dr_f'),
            Real(10 ** -5, 10 ** -1, name='learning_rate')
        ]

        return self.search_space

    def add_params(self, **params):
        pass