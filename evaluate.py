import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from SETTINGS import DEVICE, BATCH_SIZE
from datasets import ValDataset
from preprocessing import create_imagenet_preprocessing
# from default_baseline import model as default_baseline_model
from efficientnet_baseline import model as efficientnet_baseline_model
from utils.utils import convert_imagenet_to_cat_dog_naive, convert_imagenet_to_cat_dog_sophisticated

val_dataset = ValDataset('data/split/val', preprocessing=create_imagenet_preprocessing())
test_dataset = ValDataset('data/split/test', preprocessing=create_imagenet_preprocessing())

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def evaluate(model, dataloader: DataLoader, prediction_preprocessing):
    preds_all, labels_all = [], []

    for batch_x, batch_labels in tqdm(dataloader, desc="Evaluating batches.."):
        preds = model(batch_x.to(DEVICE)).cpu()
        preds = [prediction_preprocessing(p) for p in preds]

        preds_all.extend(preds)
        labels_all.extend(batch_labels)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    acc = (preds_all == labels_all).sum() / len(preds_all)

    return acc


def run_evaluation(experiment_name: str, model, prediction_preprocessing):
    val_acc = evaluate(model, val_dataloader, prediction_preprocessing)
    test_acc = evaluate(model, test_dataloader, prediction_preprocessing)

    print(f"Validation accuracy for {experiment_name}: {val_acc:.3f}")
    print(f"Test accuracy for {experiment_name}: {test_acc:.3f}")
    with open('results/metrics.csv', 'at') as f:
        f.write(f'{experiment_name},{val_acc},{test_acc}\n')


if __name__ == '__main__':
    # run_evaluation('imagenet_default_baseline', default_baseline_model, convert_imagenet_to_cat_dog_naive)
    # run_evaluation('imagenet_default_baseline_direct_probabilities', default_baseline_model,
    #                convert_imagenet_to_cat_dog_sophisticated)
    run_evaluation('efficientnetb7_baseline', efficientnet_baseline_model, convert_imagenet_to_cat_dog_sophisticated)

# TODO
#  - [+] try EfficientNet baseline (https://github.com/lukemelas/EfficientNet-PyTorch)
#  - try MobileNetV2 training baseline (no pre-trained weights)
#  - try fine-tuning with whichever baseline is better
#  - metric learning
