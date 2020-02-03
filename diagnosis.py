from math import isclose
import csv
import matplotlib.pyplot as plt


class diagnosis:
    def __init__(self):
        self.issues = []
        self.epsilon_1 = 0.35
        self.epsilon_2 = 0.45
        self.weight = 0.6

    def reset_diagnosis(self):
        self.issues = []

    def setting_overfitting_tollerance(self, e):
        self.epsilon_1 = e

    def smooth(self, scalars):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            # Calculate smoothed value
            smoothed_val = last * self.weight + (1 - self.weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def diagnosis(self, history, score, diagnosis_logs, from_):
        """
        this function take history and result of the model for make a diagnosis
        all detected problems are stored into "issues" list
        :return: list of issues
        """

        # Overfitting | Underfitting -----------------------------------------------------------------------------------

        last_training_acc = history['accuracy'][len(history['accuracy']) - 1]
        last_training_loss = history['loss'][len(history['loss']) - 1]

        if from_ == "real-time":
            self.setting_overfitting_tollerance(0.55)
            if abs(last_training_acc - score[1]) > self.epsilon_1:
                self.issues.append("gap_tr_te_acc")
            elif abs(last_training_loss - score[0]) > self.epsilon_1:
                self.issues.append("gap_tr_te_loss")
            else:
                pass

        else:
            self.setting_overfitting_tollerance(self.epsilon_1)
            if abs(last_training_acc - score[1]) > self.epsilon_1:
                self.issues.append("gap_tr_te_acc")
            elif abs(last_training_loss - score[0]) > self.epsilon_1:
                self.issues.append("gap_tr_te_loss")

        if abs(last_training_acc - 1) > self.epsilon_2:
            self.issues.append("low_acc")
        elif not isclose(last_training_loss, 0, abs_tol=0.5):
            self.issues.append("high_loss")
        else:
            pass

        # Increasing loss trend ----------------------------------------------------------------------------------------

        smoothed_loss = self.smooth(history['loss'])
        up = []
        for e in range(int(len(smoothed_loss) - 1)):
            # check growing trend
            if smoothed_loss[e] < smoothed_loss[e + 1]:
                up.append(1)

        growing = (int(len(up)) * 100) / len(history['loss'])
        if growing > 50:
            self.issues.append("growing_loss_trend")

        # Decreasing accuracy trend ------------------------------------------------------------------------------------
        # not implemented in diagnosis and tuning
        smoothed_acc = self.smooth(history['accuracy'])
        up = []
        for e in range(int(len(smoothed_acc) - 1)):
            # check growing trend
            if smoothed_acc[e] < smoothed_acc[e + 1]:
                up.append(1)

        growing = (int(len(up)) * 100) / len(history['accuracy'])
        if growing < 50:
            self.issues.append("decreasing_accuracy")

        # Floating loss ------------------------------------------------------------------------------------------------
        up = []
        down = []

        for e in range(int(len(smoothed_loss) - 1)):
            if smoothed_loss[e] < smoothed_loss[e + 1]:
                up.append(1)
            else:
                down.append(1)
        if isclose(len(up), len(down), abs_tol=150) and len(up) > 0 and len(down) > 0:
            self.issues.append("up_down_loss")

        '''
        some diagnosis to be implemented
        '''
        diagnosis_logs.write(str(self.issues))
        print(" I've found: " + str(self.issues) + "\n")
        return self.issues


if __name__ == '__main__':
    y = []
    with open('test_loss.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(float(row[2]))

    d = diagnosis()
    res = d.smooth(y)

    up = []
    down = []

    for e in range(int(len(res) - 1)):
        if res[e] < res[e + 1]:
            up.append(1)
        else:
            down.append(1)
    if isclose(len(up), len(down), abs_tol=150) and len(up) > 0 and len(down) > 0:
        print("floating finded")
