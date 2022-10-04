from torchvision.datasets import ImageFolder
import torch
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheDataset
import torchvision.transforms as transforms

domains = ['MNIST', 'MNIST_M', 'SVHN', 'SYN']
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                           transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01,
                                                                  hue=0.01),
                                           transforms.ToTensor(),
                                           transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))])
test_transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])

class DigitsSet(ImageFolder):
    def __init__(self, *args, **kwargs):
        super(DigitsSet, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def get_train_val(src, avalanche=True):
    train_sets = []
    val_sets = []
    for i in domains:
        train_sets.append(DigitsSet(root = src + 'train/' + i, transform=train_transform))
        val_sets.append(DigitsSet(root = src + 'val/' + i, transform=test_transform))
    if avalanche:
        avalanche_train_sets = []
        avalanche_val_sets = []
        for i in range(4):
            avalanche_train_sets.append(AvalancheDataset(train_sets[i], task_labels=i))
            avalanche_val_sets.append(AvalancheDataset(val_sets[i]))
        return avalanche_train_sets, avalanche_val_sets
    else:
        return train_sets, val_sets

