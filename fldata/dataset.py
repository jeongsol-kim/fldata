'''
Define basic/custom torch dataset.
'''
from torchvision import datasets
from torchvision.datasets import VisionDataset


__DATASET__ = {}


def register_dataset(name: str) -> None:
    def wrapper(cls):
        if __DATASET__.get(name):
            raise NameError(f"Name {name} is already registered.")
        __DATASET__[name] = cls
        return cls
    return wrapper


# TODO: generalize to other type of datasets including audio, video.    
def get_dataset(name: str) -> VisionDataset:
    if not __DATASET__.get(name, None):
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name]


# Centralized datasets
@register_dataset(name='base')
class CentralizedDataset(VisionDataset):
    def __init__(self, root: str):
        super().__init__(root=root)
    

@register_dataset(name='mnist')
class CentralizedMNISTDataset(datasets.MNIST):
    def __init__(self, root: str, train: bool):
        super().__init__(root=root, train=train, download=True)

@register_dataset(name='fmnist')
class CentralizedFMNISTDataset(datasets.FashionMNIST):
    def __init__(self, root: str, train: bool):
        super().__init__(root=root, train=train, download=True)

@register_dataset(name='cifar10')
class CentralizedCIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root: str, train: bool):
        super().__init__(root=root, train=train, download=True)
         


