from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from SETTINGS import CAT_LABEL, DOG_LABEL


class ValDataset(Dataset):
    def __init__(self, data_dir_path, preprocessing=None):
        self.path = Path(data_dir_path)
        self.files = list(self.path.iterdir())
        self.images = [Image.open(p) for p in tqdm(self.files, desc='Reading files')]
        self.labels = [DOG_LABEL if 'dog' in p.name else CAT_LABEL for p in self.files]

        self.preprocess = preprocessing or transforms.Compose([])  # empty function

    def __getitem__(self, item):
        return self.preprocess(self.images[item]), self.labels[item]

    def __len__(self):
        return len(self.images)
