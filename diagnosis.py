from math import isclose


class diagnosis:
    def __init__(self):
        self.issues = []
        self.epsilon_1 = 0.35
        self.epsilon_2 = 0.45
        self.weight = 0.6

    def reset_diagnosis(self):
        self.issues = []

    def smooth(self, scalars):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            # Calculate smoothed value
            smoothed_val = last * self.weight + (1 - self.weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def diagnosis(self, history, score, diagnosis_logs):
        '''
        this function take history and result of the model for make a diagnosis
        all detected problems are stored into "issues" list
        :return: list of issues
        '''

        # Overfitting | Underfitting -----------------------------------------------------------------------------------

        last_training_acc = history['accuracy'][len(history['accuracy']) - 1]
        last_training_loss = history['loss'][len(history['loss']) - 1]

        if abs(last_training_acc - score[1]) > self.epsilon_1 or abs(
                last_training_loss - score[0]) > self.epsilon_1:
            self.issues.append("overfitting")
        if abs(history['loss'][len(history['val_loss']) - 1] - 1) > self.epsilon_2 or abs(
                last_training_loss - 1) > self.epsilon_2:
            self.issues.append("underfitting")

        # Increasing loss trend ----------------------------------------------------------------------------------------

        smoothed_loss = self.smooth(history['loss'])
        up = []
        for e in range(int(len(smoothed_loss)-1)):
            # check growing trend
            if smoothed_loss[e] < smoothed_loss[e+1]:
                up.append(1)

        growing = (int(len(up)) * 100) / len(history['loss'])
        if growing > 50:
            self.issues.append("increasing_loss")

        # Decreasing accuracy trend ------------------------------------------------------------------------------------

        smoothed_acc = self.smooth(history['accuracy'])
        up = []
        for e in range(int(len(smoothed_acc) - 1)):
            # check growing trend
            if smoothed_acc[e] < smoothed_acc[e + 1]:
                up.append(1)

        growing = (int(len(up)) * 100) / len(history['accuracy'])
        if growing < 50:
            self.issues.append("decreasing_loss")

        # Floating loss ------------------------------------------------------------------------------------------------
        up = []
        down = []

        for e in range(int(len(smoothed_loss)-1)):
            if smoothed_loss[e] < smoothed_loss[e+1]:
                up.append(1)
            else:
                down.append(1)
        if not isclose(len(up), len(down), abs_tol=10):
            self.issues.append("floating_loss")

        '''
        some diagnosis to be implemented
        '''
        diagnosis_logs.write(str(self.issues))
        print(" I've found: " + str(self.issues) + "\n")
        return self.issues
