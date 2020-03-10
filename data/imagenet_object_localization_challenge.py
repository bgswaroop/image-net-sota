# This data is downloaded from the Kaggle imagenet challenge for 2020:
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data

import os
import shutil
from glob import glob
import pickle
import tarfile
import numpy as np
from data.base_data import BaseData
import xml.etree.ElementTree as ET
from configuration import root_data_dir
from data.data_genarator import DataGenerator
from pathlib import Path
import tqdm
import logging

logger = logging.getLogger(__name__)


class ImageNet(BaseData):
    def __init__(self):
        super().__init__()

        # visible attributes
        self.train_labels_map = None
        self.val_labels_map = None
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
        # self._configure_data_generators_from_directory()
        self._configure_data_generators_from_tar()

    def _set_dataset_paths(self):
        self._train_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/train")
        self._val_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/val")
        self._test_data_dir = self._root_data_dir.joinpath(r"ILSVRC/Data/CLS-LOC/test")
        self._train_labels_dir = self._root_data_dir.joinpath(r"ILSVRC/Annotations/CLS-LOC/train")
        self._val_labels_dir = self._root_data_dir.joinpath(r"ILSVRC/Annotations/CLS-LOC/val")
        self._data_tar_file = "/scratch/p288722/imagenet_object_localization_patched2019.tar"

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
                train_labels = pickle.load(f)
            with val_labels_file.open('rb') as f:
                val_labels = pickle.load(f)
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
            logger.info("Saving the computed label files to: {}, {}".format(train_labels_file, val_labels_file))
            with train_labels_file.open('wb+') as f:
                pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)
            with val_labels_file.open('wb+') as f:
                pickle.dump(val_labels, f, pickle.HIGHEST_PROTOCOL)

        # Map image names to class
        class_folders = glob(str(self._train_data_dir) + r"/*")
        labels_index = {item: idx for idx, item in enumerate(set(train_labels.values()))}
        self.train_labels_map = {os.path.basename(x): labels_index[os.path.basename(x)] for idx, x in
                                 enumerate(class_folders)}
        labels_index = {item: idx for idx, item in enumerate(set(val_labels.values()))}
        self.val_labels_map = {os.path.basename(x): labels_index[os.path.basename(x)] for idx, x in
                                 enumerate(class_folders)}

    def __train_validation_split(self, image_paths, train_percent, shuffle=True):
        # check the range for train percent
        assert (0 <= train_percent <= 1)
        # perform the split after random shuffle
        indexes = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indexes)
        training_indices = indexes[:round(len(indexes) * train_percent)]
        validation_indices = indexes[round(len(indexes) * train_percent):]
        training_data_paths = [image_paths[k] for k in training_indices]
        validation_data_paths = [image_paths[k] for k in validation_indices]
        return training_data_paths, validation_data_paths

    def _configure_data_generators_from_directory(self):
        """
        Configure data generator to load data from a directory
        """
        image_paths = glob(str(self._train_data_dir) + r"/*/*.JPEG")
        training_data_paths, validation_data_paths = self.__train_validation_split(image_paths, train_percent=0.9)

        self.train_data_gen = DataGenerator(training_data_paths, self.train_labels_map, batch_size=128, dim=(224, 224),
                                            n_channels=3, n_classes=1000, shuffle=True)
        self.val_data_gen = DataGenerator(validation_data_paths, self.train_labels_map, batch_size=128, dim=(224, 224),
                                          n_channels=3, n_classes=1000, shuffle=True)
        # todo: test data generation

    def _configure_data_generators_from_tar(self):
        """ WARNING: This operation generates a hdf5 file with size ~1.25 times the size of the original dataset. May
            result in out of disk space. This is a onetime operation. The benefit of such a transformation will result
            in quicker read and write times, especially when there are a large number of files like in ImageNet
            (~15M Images + Annotations). https://realpython.com/storing-images-in-python/
        """

        # Read the members from tar file
        tar_cache_filename = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(
            r"cache/data_tarfile_members.pkl")
        if tar_cache_filename.exists():
            with open(str(tar_cache_filename), 'rb') as f:
                tar_members = pickle.load(f)
        else:
            tar = tarfile.open(self._data_tar_file)
            tar_members = tar.getmembers()
            with open(str(tar_cache_filename), 'wb+') as f:
                pickle.dump(tar_members, f, pickle.HIGHEST_PROTOCOL)

        training_data_paths = [x for x in tar_members if "train" in x.name and ".JPEG" in x.name]
        # validation_data_paths = [x.name for x in tar_members if "val" in x.name and ".JPEG" in x.name]
        # test_data_paths = [x.name for x in tar_members if "test" in x.name and ".JPEG" in x.name]
        training_data_paths, validation_data_paths = self.__train_validation_split(training_data_paths,
                                                                                   train_percent=0.9)

        self.train_data_gen = DataGenerator(training_data_paths, self.train_labels_map, batch_size=128, dim=(224, 224),
                                            n_channels=3, n_classes=1000, shuffle=True, tar_file=self._data_tar_file)
        self.val_data_gen = DataGenerator(validation_data_paths, self.train_labels_map, batch_size=128, dim=(224, 224),
                                          n_channels=3, n_classes=1000, shuffle=True, tar_file=self._data_tar_file)
        # todo: test data generation

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
        val_labels_file = Path(os.path.dirname(os.path.realpath(__file__))).joinpath(r"cache/val_labels.pkl")
        with val_labels_file.open('rb') as f:
            val_labels = pickle.load(f)
        labels = set(val_labels.values())
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
