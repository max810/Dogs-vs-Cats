from torchvision import transforms
from torchvision.models import resnext101_32x8d

from SETTINGS import DEVICE
from preprocessing import create_imagenet_preprocessing
from utils.utils import convert_imagenet_to_cat_dog_sophisticated


def load_model():
    model = resnext101_32x8d(pretrained=True, progress=True).eval().to(DEVICE)

    return model


_preproc = create_imagenet_preprocessing()


def preprocess(img):
    return _preproc(img)


output_conversion = convert_imagenet_to_cat_dog_sophisticated
name = 'imagenet_default_baseline'
