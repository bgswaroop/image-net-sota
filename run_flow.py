import utils.session_setup
from data.imagenet_object_localization_challenge import ImageNet
from cnn_models.alex_net_vanilla import AlexNetVanilla
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info("Begin loading of ImageNet data")
    image_net_data = ImageNet()
    logger.info("End loading of ImageNet data")

    alex_net = AlexNetVanilla(image_net_data)

    alex_net.prepare_model()
    alex_net.compile_model()
    alex_net.load_model()   # to continue training
    alex_net.train_model()
