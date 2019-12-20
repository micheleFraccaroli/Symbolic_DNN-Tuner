class diagnosis:
    def __init__(self, history):
        self.history = history
        self.issues = []
        self.epsilon_1 = 0.35
        self.epsilon_2 = 0.45

    def diagnosis(self):
        '''
        this function take history and result of the model for make a diagnosis
        all detected problems are stored into "issues" list
        '''

        # overfitting | underfitting -----------------------------------------------------------------------------------

        last_training_acc = self.history['acc'][len(self.history['acc'])] - 1
        last_training_loss = self.history['loss'][len(self.history['loss'])] - 1

        if abs(last_training_acc - self.score[1]) > self.epsilon_1 or abs(
                last_training_loss - self.score[0]) > self.epsilon_1:
            self.issues.append("overfitting")
        elif abs(last_training_acc - 1) > self.epsilon_2 or abs(last_training_loss - 1) > self.epsilon_2:
            self.issues.append("underfitting")

        # other diagnosis ----------------------------------------------------------------------------------------------

        '''
        some diagnosis to be implemented
        '''
        
        return self.issues