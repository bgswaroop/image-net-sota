# This data is downloaded from the Kaggle imagenet challenge for 2020:
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data

import os
import shutil
from glob import glob
import pickle
import cv2
import tarfile
import numpy as np
from data.base_data import BaseData
import xml.etree.ElementTree as ET
from configuration import root_data_dir
from data.data_genarator import DataGenerator
from pathlib import Path
import logging
import tqdm

logger = logging.getLogger(__name__)


class ImageNet(BaseData):
    def __init__(self):
        super().__init__()

        # visible attributes
        self.train_labels = None
        self.val_labels = None
        self.train_data_gen = None
        self.val_data_gen = None
        self.test_data_gen = None

        # paths
        self._root_data_dir = root_data_dir
        self._train_data_dir = None
        self._val_data_dir = None
        self._test_data_dir = None
        self._train_labels_dir = None
        self._val_labels_dir = None
        self._data_tar_file = None

        # initialization flow
        self._set_dataset_paths()
        self._extract_image_labels()
        self._transform_val_dir_like_train_dir_hierarchy()
        self._configure_data_generators()

    def _set_dataset_paths(self):
        self._train_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/train")
        self._val_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/val")
        self._test_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/test")
        self._train_labels_dir = self._root_data_dir.joinpath(r"ILSVRC/Annotations/CLS-LOC/train")
        self._val_labels_dir = self._root_data_dir.joinpath(r"ILSVRC/Annotations/CLS-LOC/val")
        self._data_tar_file = None

    def _extract_image_labels(self):
        """
        Assumption: images contain one or more objects of the same type, therefore each image is labelled
        with the label of the first annotated object.
        Here, we extract labels for both the training and the validation parts of the data.
        """

        train_labels_file = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(r"cache/train_labels.pkl")
        val_labels_file = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(r"cache/val_labels.pkl")

        if train_labels_file.exists() and val_labels_file.exists():
            logger.info("Loading pre-computed label files: {}, {}".format(train_labels_file, val_labels_file))
            with train_labels_file.open('rb') as f:
                self.train_labels = pickle.load(f)
            with val_labels_file.open('rb') as f:
                self.val_labels = pickle.load(f)
        else:
            logger.info("Computing the image_name -> ground_truth mapping")
            train_file_names = glob(str(self._train_labels_dir) + r"/*/*")
            val_file_names = glob(str(self._val_labels_dir) + r"/*")
            train_labels = {}
            val_labels = {}

            for idx, (labels, file_names) in enumerate([(train_labels, train_file_names), (val_labels, val_file_names)]):
                progress = tqdm.tqdm(desc="LabelSet{}".format(idx), total=len(file_names), mininterval=60)
                for file_name in file_names:
                    root = ET.parse(file_name).getroot()
                    for idx, element in enumerate(root):
                        if 'object' in element.tag:
                            label = root[idx][0].text  # value
                            filename = os.path.basename(file_name.split('.')[0])  # key
                            labels[filename] = label
                    progress.update(1)

            # Labels are dictionaries, with image name (without extension) as their key,
            # mapped to the corresponding labels as their value.
            self.train_labels = train_labels
            self.val_labels = val_labels

            logger.info("Saving the computed label files to: {}, {}".format(train_labels_file, val_labels_file))
            with train_labels_file.open('wb+') as f:
                pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)
            with val_labels_file.open('wb+') as f:
                pickle.dump(val_labels, f, pickle.HIGHEST_PROTOCOL)

    def _configure_data_generators(self):
        image_paths = glob(str(self._train_data_dir) + r"*/*.JPEG")
        class_folders = glob(str(self._train_data_dir) + r"/*")
        labels_index = {item: idx for idx, item in enumerate(set(self.train_labels.values()))}
        labels_map = {os.path.basename(x): labels_index[os.path.basename(x)] for idx, x in enumerate(class_folders)}

        training_data_split = 0.9  # value between 0 and 1
        assert (0 <= training_data_split <= 1)
        indexes = np.arange(len(image_paths))
        np.random.shuffle(indexes)
        training_indices = indexes[:round(len(indexes) * training_data_split)]
        validation_indices = indexes[round(len(indexes) * training_data_split):]

        training_data_paths = [image_paths[k] for k in training_indices]
        validation_data_paths = [image_paths[k] for k in validation_indices]
        self.train_data_gen = DataGenerator(training_data_paths, labels_map, batch_size=16, dim=(224, 224),
                                            n_channels=3, n_classes=1000, shuffle=True)
        self.val_data_gen = DataGenerator(validation_data_paths, labels_map, batch_size=16, dim=(224, 224),
                                          n_channels=3, n_classes=1000, shuffle=True)
        # todo: go over the images in the test dir and generate ids (image paths)

    # todo: Loading the data from a single tar.gz file (future work!)
    def __transform_data_into_hdf5(self):
        """ WARNING: This operation generates a hdf5 file with size ~1.25 times the size of the original dataset. May
            result in out of disk space. This is a onetime operation. The benefit of such a transformation will result
            in quicker read and write times, especially when there are a large number of files like in ImageNet
            (~15M Images + Annotations). https://realpython.com/storing-images-in-python/
        """

        if os.path.exists('cache/data_tarfile_members.pkl'):
            with open('cache/data_tarfile_members.pkl', 'rb') as f:
                q = pickle.load(f)

        else:
            tar = tarfile.open(self._data_tar_file)
            q = tar.getmembers()
            with open('cache/data_tarfile_members.pkl', 'wb+') as f:
                pickle.dump(q, f, pickle.HIGHEST_PROTOCOL)

        # with tarfile.open(self.__data_tar_file) as tar:
        #     l = []
        #     idx = 0
        #     for member in tar:
        #         if idx == 10:
        #             break
        #         idx += 1
        #         l.append(member)

        # Step 1: Read all the images using OpenCV (Among all the read methods that I know, OpenCV is the quickest)
        train_image_paths = glob(str(self._train_data_dir) + r"/*/*")

        # Step 2: Preprocess the images according to the AlexNet requirements
        for idx, image_path in enumerate(train_image_paths):
            img = cv2.imread(image_path)
            img = self.__preprocess_inputs(img)

    def _transform_val_dir_like_train_dir_hierarchy(self):
        """
        Transforms the val_dir into the following hierarchy:
        val (dir)
            - class1 (dir)
                - image1
                - image2
                - ...
            - class2 (dir)
                - image1
                - image2
                - ...
            - ...
        This transformation will allow the use of keras image data generator API.
        :return: None
        """

        image_files = glob(str(self._val_data_dir) + r"/*.JPEG")

        # 1. Create class wise folders
        if self.val_labels is None:
            self._extract_image_labels()
        labels = set(self.val_labels.values())
        for label in labels:
            os.makedirs(str(self._val_data_dir.joinpath(label)), exist_ok=True)

        # 2. move the val images into the corresponding class wise folder
        for image_file_path in image_files:
            image_name = os.path.basename(image_file_path.split('.')[0])
            destination_dir = self._val_data_dir.joinpath(self.val_labels[image_name])
            shutil.move(src=image_file_path, dst=str(destination_dir))

    # # The overloaded method (ofourse cant use it in python simultaneously), used keras in-built data generators
    # def __configure_data_generators(self):
    #
    #     datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(
    #         rescale=1. / 255,
    #         horizontal_flip=True,  # data augmentation
    #         validation_split=0.1,  # 10 percent of all training images will be used for validation
    #     )
    #
    #     # 90% of train data for "training"
    #     self.train_data_gen = datagen1.flow_from_directory(
    #         self.__train_data_dir,
    #         target_size=(224, 224),
    #         batch_size=16,
    #         seed=123,  # using a fixed seed to generate reproducible results
    #         subset="training"
    #     )
    #
    #     # 10% of train data for "validation"
    #     self.val_data_gen = datagen1.flow_from_directory(
    #         self.__train_data_dir,
    #         target_size=(224, 224),
    #         batch_size=16,
    #         seed=123,  # using a fixed seed to generate reproducible results
    #         subset="validation"
    #     )
    #
    #     datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(
    #         rescale=1. / 255,
    #     )
    #
    #     self.test_data_gen = datagen2.flow_from_directory(
    #         self.__val_data_dir,
    #         target_size=(256, 256),
    #         batch_size=1,
    #         seed=123,  # To generate samples for prediction in a fixed order
    #     )


if __name__ == "__main__":
    data = ImageNet()
    pass
