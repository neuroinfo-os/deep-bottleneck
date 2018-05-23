from tensorflow.python.keras.callbacks import Callback


def load(monitor='val_acc', value=0.94):
    return EarlyStoppingAtSpecificAccuracy(monitor, value)


class EarlyStoppingAtSpecificAccuracy(Callback):


    @classmethod
    def load(cls,  monitor='val_acc', value=0.94):
        return cls(monitor, value)


    def __init__(self, monitor, value):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        print('current = ', current)
        print('Defined value: ', self.value)

        if current > self.value:
            print('Current is higher than the defined value. Current = ', current, ', value = ', self.value)
            self.model.stop_training = True