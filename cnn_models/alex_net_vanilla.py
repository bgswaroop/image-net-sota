import os
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau
from utils.callbacks import SaveModelOnEpochEnd, LogLossOnBatchEnd
from cnn_models.base_model import BaseModel
from data.imagenet_object_localization_challenge import ImageNet
import pickle
from utils.session_setup import SessionSetup
import logging

logger = logging.getLogger(__name__)


class AlexNetVanilla(BaseModel):
    def __init__(self, data):
        super().__init__(data)
        self.cnn_input_dims = (224, 224, 3,)
        self.weight_decay = 0.0005
        self.start_epoch = 0
        self.end_epoch = 90

    def prepare_model(self):
        # Inputs
        inputs = tf.keras.Input(shape=self.cnn_input_dims, dtype='float32')

        # Layers 1 and 2 are convolutional layers are response normalized layers with max-pool
        x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), activation="relu", strides=(4, 4),
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation="relu",
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                   bias_initializer=tf.constant_initializer(1))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)

        # Layers 3, 4 and 5 are convolutional layers with a single max-pooled layer at the end
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu",
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                   bias_initializer=tf.constant_initializer(1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                   bias_initializer=tf.constant_initializer(1))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)

        # Layers 6, 7 and 8 are dense layers
        x = tf.keras.layers.Dense(units=4096, activation="relu",
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                  kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Dense(units=4096, activation="relu",
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                  kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        outputs = tf.keras.layers.Dense(units=1000, activation="softmax",
                                        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="AlexNetVanilla")

    def compile_model(self):
        """
        Configures the model for training
        """

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            # optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5), "acc"]
        )
        self.model.summary(print_fn=logger.info)

    def train_model(self):
        """
        Perform the training of the Alex-net model
        """
        runtime_data_dir = str(SessionSetup().get_session_folder_path())

        train_history = self.model.fit(
            x=self.data.train_data_gen,
            epochs=self.end_epoch,
            verbose=2,
            callbacks=[SaveModelOnEpochEnd(runtime_data_dir, 10), LogLossOnBatchEnd(runtime_data_dir),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, verbose=1)],
            validation_data=self.data.val_data_gen,
            shuffle=False,  # This parameter may cause inconsistency between multiple runs with same inputs
            initial_epoch=self.start_epoch,  # Change this parameter when training is resumed
            validation_freq=1,
        )

        # save the training history
        filename = "train_history_epoch_{}_to_{}.pkl".format(self.start_epoch, self.end_epoch)
        full_file_path = os.path.join(runtime_data_dir, filename)
        with open(full_file_path, 'wb') as f:
            pickle.dump(train_history.history, f)

        return train_history

    def evaluate_model(self, input_img):
        """
        At test time, the network makes a prediction by extracting
        five 224 x 224 patches (the four corner patches and the center patch) as well as their horizontal
        reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax
        layer on the ten patches.
        """

        image_batch = [input_img[:224, :224], np.fliplr(input_img[:224, :224]),
                       input_img[-224:, :224], np.fliplr(input_img[-224:, :224]),
                       input_img[:224, -224:], np.fliplr(input_img[:224, -224:]),
                       input_img[-224:, -224:], np.fliplr(input_img[-224:, -224:]),
                       input_img[16:240, 16:240], np.fliplr(input_img[16:240, 16:240])]

        batch_predictions = self.model.predict(image_batch)

        # average the batch predictions
        # return the top 1 and top 5 predictions


if __name__ == "__main__":
    image_net_data = ImageNet()
    alex_net = AlexNetVanilla(image_net_data)

    alex_net.prepare_model()
    alex_net.compile_model()
    alex_net.train_model()
