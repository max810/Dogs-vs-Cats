from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from SETTINGS import CAT_LABEL, DOG_LABEL


# AVERAGE HEIGHT: 417 px
# AVERAGE WIDTH:  467 px

class CatDogDataset(Dataset):
    def __init__(self, data_dir_path, preprocessing=None, with_files=False, with_preproc=True):
        self.path = Path(data_dir_path)
        self.files = list(self.path.iterdir())
        self.images = []
        self.with_files = with_files
        self.with_preproc = with_preproc
        for p in tqdm(self.files, desc='Reading files'):
            with Image.open(p) as im:
                self.images.append(im.copy())
        self.labels = [DOG_LABEL if 'dog' in p.name else CAT_LABEL for p in self.files]

        self.preprocess = preprocessing or transforms.Compose([])  # empty function

    def _return_items(self, item):
        if self.with_preproc:

            return self.preprocess(self.images[item]), self.labels[item]
        else:
            return self.images[item], self.labels[item]

    def _return_items_with_files(self, item):
        return self.files[item].name, *self._return_items(item)

    def __getitem__(self, item):
        if self.with_files:
            return self._return_items_with_files(item)
        else:
            return self._return_items(item)

    def __len__(self):
        return len(self.images)


class InferenceCatDogDataset(Dataset):
    def __init__(self, data_dir_path, preprocessing=None, with_files=False, with_preproc=True):
        self.path = Path(data_dir_path)
        self.files = list(self.path.iterdir())
        self.images = []
        self.with_files = with_files
        self.with_preproc = with_preproc
        for p in tqdm(self.files, desc='Reading files'):
            with Image.open(p) as im:
                self.images.append(im.copy())
        self.preprocess = preprocessing or transforms.Compose([])  # empty function

    def _return_items(self, item):
        if self.with_preproc:

            return self.preprocess(self.images[item])
        else:
            return self.images[item]

    def _return_items_with_files(self, item):
        return self._return_items(item), self.files[item].name,

    def __getitem__(self, item):
        if self.with_files:
            return self._return_items_with_files(item)
        else:
            return self._return_items(item)

    def __len__(self):
        return len(self.images)
