import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import random_split

cwd = os.getcwd()
directory_clean = cwd + "\\plates\\train\\cleaned"
directory_dirty = cwd + "\\plates\\train\\dirty"

pilToTensor = transforms.ToTensor()
tensorToPil = transforms.ToPILImage()


def tensors_from_images(folder, label):
    tensors = []
    for root, directories, files in os.walk(folder):
        for file in files:
            img = Image.open(root + '\\' + file)
            tensors.append([pilToTensor(img), label])
    print(tensors)
    return tensors


def shuffle(dataset, ratio=0.8):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, validation_dataset


dataset = tensors_from_images(directory_clean, 0) + tensors_from_images(directory_dirty, 1)
train_dataset, validation_dataset = shuffle(dataset)
print(len(train_dataset))
print(len(validation_dataset))
print(train_dataset)
