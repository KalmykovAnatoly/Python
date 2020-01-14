import os
from random import randrange

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import random_split

cwd = os.getcwd()
directory_clean = cwd + "\\plates\\train\\cleaned"
directory_dirty = cwd + "\\plates\\train\\dirty"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def tensors_from_images(folder, label, transform):
    tensors = []
    for root, directories, files in os.walk(folder):
        for file in files:
            img = Image.open(root + '\\' + file)
            tensors.append([transform(img), label])
    return tensors


def shuffle(dataset, ratio=0.8):
    """

    :type dataset: list
    :type ratio: float
    """
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, validation_dataset


def check_random_out(dataset):
    """

    :type dataset: list
    """
    tensor_to_pil = transforms.ToPILImage()
    plt.imshow(tensor_to_pil(train_dataset[randrange(len(dataset))][0]))
    plt.show()


dataset = tensors_from_images(directory_clean, 0, preprocess) + tensors_from_images(directory_dirty, 1, preprocess)
train_dataset, validation_dataset = shuffle(dataset)
check_random_out(train_dataset)
