import logging

logger = logging.getLogger(__name__)


class BaseData:
    def __init__(self):
        self._root_data_dir = None

        # visible attributes
        self.train_labels = None
        self.val_labels = None
        self.train_data_gen = None
        self.val_data_gen = None
        self.test_data_gen = None

    def _set_dataset_paths(self):
        pass
