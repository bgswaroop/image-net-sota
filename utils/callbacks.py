import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class LogLossOnBatchEnd(tf.keras.callbacks.Callback):
    def __init__(self, results_path, num_batches=25):
        self.results_path = results_path
        self.num_batches = num_batches
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if batch != 0 and batch % self.num_batches == 0:
            if not os.path.exists(self.results_path):
                os.mkdir(self.results_path)
            logger.info("batch {} ".format(batch) + str(logs))


class SaveModelOnEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, results_path, num_epochs=1):
        super().__init__()
        self.results_path = results_path
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.num_epochs == 0:
            if not os.path.exists(self.results_path):
                os.mkdir(self.results_path)

            self.model.save(
                self.results_path + os.path.sep + "epoch{}_acc{}_loss{}_valAcc{}_valLoss{}.h5".format(
                    epoch,
                    str(round(logs['acc'], 2)),
                    str(round(logs['loss'], 2)),
                    str(round(logs['val_acc'], 2)),
                    str(round(logs['val_loss'], 2)))
            )

            logger.info("Saved the intermediate model for epoch number : {}".format(epoch))
        logger.info("Learning rate: {}".format(float(tf.keras.backend.get_value(self.model.optimizer.lr))))
