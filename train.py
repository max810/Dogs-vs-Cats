import shutil
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from tqdm import tqdm
import matplotlib.pyplot as plt

from SETTINGS import DEVICE, BATCH_SIZE
from datasets import CatDogDataset
from preprocessing import create_imagenet_preprocessing, create_fast_preprocessing_with_augs, create_fast_preprocessing


def train_model(model: nn.Module, dataloaders: Dict, criterion, optimizer, experiment_name, num_epochs):
    p_weights = Path(f'tmp/{experiment_name}/model_weights/')
    p_plots = Path(f'tmp/{experiment_name}/plots/')

    if not p_weights.exists():
        p_weights.mkdir(parents=True)
    if not p_plots.exists():
        p_plots.mkdir(parents=True)

    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_epoch_stats = {}

    for epoch in range(1, num_epochs + 1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'Epoch {epoch}, {phase}...'):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.cpu().item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).double().cpu()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, f'{p_weights}/{epoch}.pt')
                best_epoch_stats['time'] = time.time() - since
                best_epoch_stats['train_acc'] = train_acc_history[-1]
                best_epoch_stats['val_acc'] = epoch_acc

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

                plt.figure()
                plt.plot(train_acc_history, label='Train acc')
                plt.plot(val_acc_history, label='Val acc')
                plt.xticks(list(range(1, epoch + 1)), rotation=-60)
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig(p_plots / 'accuracy.png')

                plt.figure()
                plt.plot(train_loss_history, label='Train loss')
                plt.plot(val_loss_history, label='Val loss')
                plt.xticks(list(range(1, epoch + 1)), rotation=-60)
                plt.xlabel('Epoch')
                plt.savefig(p_plots / 'loss.png')
                plt.legend()

                plt.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model = torch.load(f'{p_weights}/{best_epoch}.pt')
    shutil.copyfile(f'{p_weights}/{best_epoch}.pt',
                    f'{p_weights.parent}/BEST_{best_epoch}.pt')

    with open('results/training_metrics.csv', 'at') as f:
        f.write(
            f'{experiment_name},{best_epoch_stats["train_acc"]},'
            f'{best_epoch_stats["val_acc"]},{best_epoch},{best_epoch_stats["time"]}\n')

    return model, val_acc_history


if __name__ == '__main__':
    model = mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 2),
    )

    model.to(DEVICE)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

    train_dataset = CatDogDataset('data/split/train', preprocessing=create_fast_preprocessing_with_augs())
    val_dataset = CatDogDataset('data/split/val', preprocessing=create_fast_preprocessing())

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    train_model(model, {'train': train_dataloader, 'val': val_dataloader}, loss, optimizer,
                'mobilenet_v2_pretrained_simpleaug', 200)
