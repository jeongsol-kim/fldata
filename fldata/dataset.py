'''
Define basic/custom torch dataset.
'''
import numpy as np
from functools import partial
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

# TODO: generalize to other type of datasets includign audio, video
def load_data(name: str) -> np.ndarray:
    dataset = get_dataset(name)
    
    return dataset.images, dataset.labels



# Centralized datasets
@register_dataset(name='centre_base')
class CentralizedDataset(VisionDataset):
    def __init__(self):
        super().__init__()
    

@register_dataset(name='centre_mnist')
class CentralizedMNISTDataset(datasets.MNIST):
    def __init__(self, root):
        super().__init__(root, download=True)
