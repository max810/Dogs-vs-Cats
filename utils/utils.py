import torch

from SETTINGS import CAT_LABEL, DOG_LABEL

DOG_FROM = 151
DOG_TO = 276

CAT_FROM = 281
CAT_TO = 293


def _convert(idx):
    if _is_dog(idx):
        return DOG_LABEL
    else:
        return CAT_LABEL


def convert_imagenet_to_cat_dog_naive(logprobs):
    return _convert(logprobs.argmax(axis=0))


def _is_dog(idx):
    return DOG_FROM <= idx <= DOG_TO


def _is_cat(idx):
    return CAT_FROM <= idx <= CAT_TO


def convert_imagenet_to_cat_dog_sophisticated(probs):
    """
    We ignore probabilities for other classes and compare cats to dogs directly
    :param probs:
    :return:
    """
    dog_cat_logprobs = {idx: val for idx, val in enumerate(probs) if
                        _is_dog(idx) or _is_cat(idx)}

    max_idx = max(dog_cat_logprobs.keys(), key=(lambda key: dog_cat_logprobs[key]))

    return _convert(max_idx)


def convert_binary_logits_to_cat_dog(probs):
    class_idx = torch.argmax(probs)
    return int(class_idx)
