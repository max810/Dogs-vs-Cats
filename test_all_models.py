import pandas as pd
import torch
from PIL import Image

import models.default_baseline as default_baseline

import models.efficientnet_baseline as efficientnet_baseline
import models.mobilenet_local as mobilenet_local
import models.mobilenet_colab as mobilenet_colab
import models.resnext_colab as resnext_colab

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from SETTINGS import DEVICE, BATCH_SIZE
from datasets import CatDogDataset, InferenceCatDogDataset
from preprocessing import tta


def predict(model, dataloader: DataLoader, prediction_preprocessing):
    preds_all, files_all = [], []

    for batch_x, batch_files in tqdm(dataloader, desc="Evaluating batches.."):
        preds = model(batch_x.to(DEVICE)).cpu()
        preds = [prediction_preprocessing(p) for p in preds]

        preds_all.extend(preds)
        files_all.extend(batch_files)

    preds_all = np.array(preds_all)

    res = pd.DataFrame({'filename': files_all, 'pred': preds_all.astype(bool)})

    return res


def evaluate(model, dataloader: DataLoader, prediction_preprocessing):
    preds_all, labels_all, files_all = [], [], []

    for batch_files, batch_x, batch_labels in tqdm(dataloader, desc="Evaluating batches.."):
        preds = model(batch_x.to(DEVICE)).cpu()
        preds = [prediction_preprocessing(p) for p in preds]

        preds_all.extend(preds)
        labels_all.extend(batch_labels)
        files_all.extend(batch_files)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    acc = (preds_all == labels_all).sum() / len(preds_all)

    trues = ['dog' in f for f in files_all]

    res = pd.DataFrame({'filename': files_all, 'true': trues, 'pred': preds_all.astype(bool)})

    return acc, res


def evaluate_tta(model, dataset: CatDogDataset, prediction_preprocessing):
    preds_all, labels_all, files_all = [], [], []

    for file, x, y in tqdm(dataset, desc="Evaluating batches.."):
        batch = torch.stack([dataset.preprocess(i) for i in tta(x)])

        preds = model(batch.to(DEVICE)).cpu()
        preds = [prediction_preprocessing(p) for p in preds]

        preds_all.append(np.mean(preds))
        labels_all.append(y)
        files_all.append(file)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    acc = (preds_all == labels_all).sum() / len(preds_all)

    trues = ['dog' in f for f in files_all]

    res = pd.DataFrame({'filename': files_all, 'true': trues, 'pred': preds_all.astype(bool)})

    return acc, res


def run_prediction(experiment_name: str, model, prediction_preprocessing, dataloader):
    preds_df = predict(model, dataloader, prediction_preprocessing)
    preds_df.to_csv(f'results/{experiment_name}_test_preds_PREDS_ONLY.csv', index=False)


def run_evaluation(experiment_name: str, model, prediction_preprocessing, dataloader):
    test_acc, test_preds_df = evaluate(model, dataloader, prediction_preprocessing)

    print(f"Test accuracy for {experiment_name}: {test_acc:.3f}")
    with open('results/test_metrics.csv', 'at') as f:
        f.write(f'{experiment_name},{test_acc}\n')

    test_preds_df.to_csv(f'results/{experiment_name}_test_preds_{test_acc:.3f}.csv', index=False)


def run_evaluation_tta(experiment_name: str, model, prediction_preprocessing, dataset):
    test_acc, test_preds_df = evaluate_tta(model, dataset, prediction_preprocessing)

    print(f"Test accuracy for {experiment_name}: {test_acc:.3f}")
    with open('results/tta_test_metrics.csv', 'at') as f:
        f.write(f'{experiment_name},{test_acc}\n')

    test_preds_df.to_csv(f'results/tta_{experiment_name}_test_preds_{test_acc:.3f}.csv', index=False)


if __name__ == '__main__':
    # sorted desc by acc
    for module in [resnext_colab, mobilenet_colab, default_baseline, mobilenet_local]:
        print(f"Testing {module.name}")

        test_dataset = InferenceCatDogDataset('data/test1/test1', preprocessing=module.preprocess, with_files=True)

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        run_prediction(module.name, module.load_model(), module.output_conversion, test_dataloader)

# TODO
#  - [+] try EfficientNet baseline (https://github.com/lukemelas/EfficientNet-PyTorch)
#  - [-] try MobileNetV2 training baseline (no pre-trained weights)
#  - [+] try fine-tuning with whichever baseline is better
#  - [-] metric learning
#  - [+] ensemble (maybe train multiple models with different augmentations AND/OR different train/test splits?)?
#  - [+] TTA, didn't work
