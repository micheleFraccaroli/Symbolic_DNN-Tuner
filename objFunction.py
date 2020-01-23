class objFunction:
    def __init__(self, search, controller):
        self.search_space = search
        self.controller = controller
        self.space = {}

    def objective(self, params):
        for i,j in zip(self.search_space, params):
            self.space[i.name] = j

        f = open("algorithm_logs/hyperparameters.txt", "a")
        to_optimize = self.controller.training(self.space)
        f.close()
        return to_optimize