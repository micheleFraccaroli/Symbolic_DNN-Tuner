import tensorflow as tf
import tensorflow.keras.backend as K


# Loss - Learning Rate curve for finding best learning rate range

class Lolr(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, steps_per_epoch):
        super().__init__()
        self.losses = []
        self.lrs = []
        self.steps_per_epoch = steps_per_epoch + 7
        #self.tot_iter = self.steps_per_epoch + epochs
        self.iter = 0
        self.new_lr = None
        self.min_lr = min_lr
        self.max_lr = max_lr


    def on_batch_end(self, batch, logs={}):
        self.iter += 1
        self.losses.append(logs.get('loss'))
        actual_lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(actual_lr)
        if actual_lr < self.max_lr:
            x = self.iter / self.steps_per_epoch
            self.new_lr = self.min_lr + (self.max_lr - self.min_lr) * x

        K.set_value(self.model.optimizer.lr, self.new_lr)
