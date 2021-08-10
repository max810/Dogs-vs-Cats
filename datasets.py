from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from SETTINGS import CAT_LABEL, DOG_LABEL


# AVERAGE HEIGHT: 417 px
# AVERAGE WIDTH:  467 px

class CatDogDataset(Dataset):
    def __init__(self, data_dir_path, preprocessing=None):
        self.path = Path(data_dir_path)
        self.files = list(self.path.iterdir())
        self.images = []
        for p in tqdm(self.files, desc='Reading files'):
            with Image.open(p) as im:
                self.images.append(im.resize((256, 256)).copy())
        self.labels = [DOG_LABEL if 'dog' in p.name else CAT_LABEL for p in self.files]

        self.preprocess = preprocessing or transforms.Compose([])  # empty function

    def __getitem__(self, item):
        return self.preprocess(self.images[item]), self.labels[item]

    def __len__(self):
        return len(self.images)
