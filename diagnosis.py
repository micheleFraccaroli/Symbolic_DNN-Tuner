import platform

class diagnosis:
    def __init__(self):
        self.issues = []
        self.epsilon_1 = 0.35
        self.epsilon_2 = 0.45

    def diagnosis(self, history, score, diagnosis_logs):
        '''
        this function take history and result of the model for make a diagnosis
        all detected problems are stored into "issues" list
        :return: list of issues
        '''

        # overfitting | underfitting -----------------------------------------------------------------------------------
        if platform.system() == 'Darwin':
            metric = 'acc'
        else:
            metric = 'accuracy'

        last_training_acc = history[metric][len(history[metric])-1]
        last_training_loss = history['loss'][len(history['loss'])-1]

        if abs(last_training_acc - score[1]) > self.epsilon_1 or abs(
                last_training_loss - score[0]) > self.epsilon_1:
            self.issues.append("overfitting")
        if abs(last_training_acc - 1) > self.epsilon_2 or abs(last_training_loss - 1) > self.epsilon_2:
            self.issues.append("underfitting")

        # other diagnosis ----------------------------------------------------------------------------------------------

        '''
        some diagnosis to be implemented
        '''

        diagnosis_logs.write(str(self.issues))
        print(" I've found: " + str(self.issues) + "\n")
        return self.issues
