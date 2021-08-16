from efficientnet_pytorch import EfficientNet

from SETTINGS import DEVICE
from preprocessing import create_efficientnet_preprocessing
from utils.utils import convert_imagenet_to_cat_dog_sophisticated


def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b5').eval().to(DEVICE)

    return model


_preproc = create_efficientnet_preprocessing()


def preprocess(img):
    return _preproc(img)


output_conversion = convert_imagenet_to_cat_dog_sophisticated
name = 'efficientnetb5_baseline'
