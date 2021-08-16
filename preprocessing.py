from torchvision.transforms import transforms


def create_imagenet_preprocessing():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing


def create_efficientnet_preprocessing():
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return tfms


def create_fast_preprocessing_with_augs():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        # transforms.Resize(150),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(0.1),
        transforms.RandomGrayscale(0.05),
        transforms.RandomEqualize(0.1),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing


def create_fast_preprocessing():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing


def tta(img):
    transforms_ = [
        transforms.RandomHorizontalFlip(p=1.0),
        # transforms.RandomRotation((-10, 10)),
        transforms.RandomResizedCrop(img.size, (0.8, 1.0)),
        # transforms.RandomEqualize(p=1.0),
        # transforms.RandomAutocontrast(p=1.0),
    ]

    return [
        img, *[t(img) for t in transforms_]
    ]
