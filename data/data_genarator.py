# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2


# todo: data augmentation
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, labels, batch_size=16, dim=(224, 224), n_channels=3, n_classes=1000, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.image_size = 224

        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_ids))
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __preprocess_inputs(self, img_path):
        """
        :param img: Any rectangular RGB image as a numpy array
        :return: Cropped and resized image with dimensions 256 x 256
        """
        # resize the input image keeping the shortest side to 256
        img = cv2.imread(img_path)
        input_img_shape = np.array([img.shape[1], img.shape[0]])
        scale_factor = self.image_size / input_img_shape.min()
        resized_img = cv2.resize(img, tuple(np.round(input_img_shape * scale_factor).astype(int)),
                                 interpolation=cv2.INTER_LINEAR)

        # crop the centered input image to (256 x 256)
        longer_side = np.array(resized_img.shape).argmax()
        start = np.round(resized_img.shape[longer_side] / 2 - self.image_size / 2).astype(int)
        end = start + self.image_size
        if longer_side == 0:
            pre_processed_img = resized_img[start:end, :]
        else:
            pre_processed_img = resized_img[:, start:end]

        return pre_processed_img

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, image_path in enumerate(list_ids_temp):
            # Store sample
            # use OpenCV to read image
            x[i,] = self.__preprocess_inputs(image_path)

            # Store class
            y[i] = self.labels[Path(image_path)._parts[-2]]

        return x, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y
