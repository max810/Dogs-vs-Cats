import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from SETTINGS import DEVICE
from utils.utils import convert_binary_logits_to_cat_dog


def load_model():
    model = torch.load('all_models/mobilenet_v2_colab_epoch_3_98_val.pt').eval().to(DEVICE)

    return model


_preproc = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess(img):
    return _preproc(img)


output_conversion = convert_binary_logits_to_cat_dog
name = 'mobilenetv2_colab'
