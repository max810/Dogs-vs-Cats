import random
import shutil
from pathlib import Path

from tqdm import tqdm

data_path = Path('data/train/train')

train_split_path = Path('data/split/train')
val_split_path = Path('data/split/val')
test_split_path = Path('data/split/test')
N_files = 25_000
N_val = 2500
N_test = 2500


def gen_files_split(orig_files, k):
    dog_files = [x for x in orig_files if 'dog' in x.name]
    cat_files = [x for x in orig_files if 'cat' in x.name]

    dog_files_split = set(random.sample(dog_files, k=k // 2))
    cat_files_split = set(random.sample(cat_files, k=k // 2))

    orig_files = (set(dog_files) - dog_files_split).union(set(cat_files) - cat_files_split)
    split_files = cat_files_split.union(dog_files_split)

    return orig_files, split_files


if __name__ == '__main__':
    random.seed(808)
    shutil.rmtree('data/split')

    orig_files = list(data_path.iterdir())
    random.shuffle(orig_files)
    train_files, val_files = gen_files_split(orig_files, k=N_val)
    train_files, test_files = gen_files_split(train_files, k=N_test)

    train_split_path.mkdir(parents=True)
    val_split_path.mkdir(parents=True)
    test_split_path.mkdir(parents=True)

    for f in tqdm(train_files, desc='Copying training files'):
        shutil.copyfile(f, train_split_path / f.name)

    for f in tqdm(val_files, desc='Copying validation files'):
        shutil.copyfile(f, val_split_path / f.name)

    for f in tqdm(test_files, desc='Copying test files'):
        shutil.copyfile(f, test_split_path / f.name)
