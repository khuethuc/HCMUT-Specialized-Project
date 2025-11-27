from torchvision import datasets, transforms

data_dir = "../data"

datasets.CIFAR10(root=data_dir, train=True, download=True)

datasets.CIFAR10(root=data_dir, train=False, download=True)