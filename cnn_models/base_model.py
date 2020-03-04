import logging

logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, data):
        self.model = None
        self.data = data

    def prepare_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self, input_img):
        pass
