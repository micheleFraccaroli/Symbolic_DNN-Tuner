from tensorflow.keras import callbacks

from diagnosis import diagnosis


class real_time_analysis(callbacks.Callback):
    def __init__(self):
        self.hist = {'loss' : [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.score = []

    def set_epochs(self, epochs):
        self.epochs = epochs

    def on_train_begin(self, logs={}):
        self.accuracy = []
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('accuracy'))
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.hist['loss'].append(logs['loss'])
        self.hist['val_loss'].append(logs['val_loss'])
        self.hist['accuracy'].append(logs['accuracy'])
        self.hist['val_accuracy'].append(logs['val_accuracy'])

        if epoch % 10 == 0 and epoch > 0:
            diagnosis_logs = open("algorithm_logs/diagnosis_logs.txt", "a")
            self.d = diagnosis()
            self.d.reset_diagnosis()
            self.score.append(self.hist['val_loss'][len(self.hist['val_loss']) - 1])
            self.score.append(self.hist['val_accuracy'][len(self.hist['val_accuracy'])-1])
            self.issues = self.d.diagnosis(self.hist, self.score, diagnosis_logs, "real-time")
            diagnosis_logs.close()