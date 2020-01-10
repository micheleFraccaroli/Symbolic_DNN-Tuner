class diagnosis:
    def __init__(self, history, diagnosis_logs, score):
        self.history = history
        self.score = score
        self.issues = []
        self.epsilon_1 = 0.35
        self.epsilon_2 = 0.45
        self.diagnosis_logs = diagnosis_logs

    def diagnosis(self):
        '''
        this function take history and result of the model for make a diagnosis
        all detected problems are stored into "issues" list
        :return: list of issues
        '''

        # overfitting | underfitting -----------------------------------------------------------------------------------

        last_training_acc = self.history['acc'][len(self.history['acc'])-1]
        last_training_loss = self.history['loss'][len(self.history['loss'])-1]

        if abs(last_training_acc - self.score[1]) > self.epsilon_1 or abs(
                last_training_loss - self.score[0]) > self.epsilon_1:
            self.issues.append("overfitting")
        if abs(last_training_acc - 1) > self.epsilon_2 or abs(last_training_loss - 1) > self.epsilon_2:
            self.issues.append("underfitting")

        # other diagnosis ----------------------------------------------------------------------------------------------

        '''
        some diagnosis to be implemented
        '''

        self.diagnosis_logs.write(str(self.issues))
        return self.issues
