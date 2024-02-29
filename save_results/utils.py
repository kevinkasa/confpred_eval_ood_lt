"""
Saves softmax results to .npz file.
"""
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch


def save_results(save_path, model, data_loader, device, exp_name):
    """
    Save softmax results and corresponding labels for a given dataset and model
    """
    scores = np.ones((len(data_loader.dataset), 1000))
    labels = np.ones((len(data_loader.dataset),))
    counter = 0
    # do inference
    with torch.no_grad():
        for batch in tqdm(data_loader):
            scores[counter:counter + batch[0].shape[0], :] = model(batch[0].to(device)).softmax(dim=1).cpu().numpy()
            labels[counter:counter + batch[1].shape[0]] = batch[1].numpy().astype(int)
            counter += batch[0].shape[0]

    print("saving the scores and labels")
    os.makedirs(save_path, exist_ok=True)
    np.savez(save_path + exp_name + '.npz', smx=scores, labels=labels)

    acc = (np.argmax(scores, axis=1) == labels).mean() * 100
    print('Validation accuracy: {} %'.format(acc))


def save_results_IN200(save_path, model, data_loader, device, exp_name, indices):
    """
    Save results for 200-class variants of IN
    """
    scores = np.ones((len(data_loader.dataset), 200))
    labels = np.ones((len(data_loader.dataset),))
    counter = 0
    # do inference
    with torch.no_grad():
        for batch in tqdm(data_loader):
            scores[counter:counter + batch[0].shape[0], :] = model(batch[0].to(device))[:, indices].softmax(
                dim=1).cpu().numpy()
            labels[counter:counter + batch[1].shape[0]] = batch[1].numpy().astype(int)
            counter += batch[0].shape[0]

    print("saving the scores and labels")
    os.makedirs(save_path, exist_ok=True)
    np.savez(save_path + exp_name + '.npz', smx=scores, labels=labels)

    acc = (np.argmax(scores, axis=1) == labels).mean() * 100
    print('Validation accuracy: {} %'.format(acc))


# set up imagenet V2 loader
class ImageNetV2(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.fnames = list(self.root.glob('**/*.jpeg'))
        self.targets = [name.parent.name for name in self.fnames]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
