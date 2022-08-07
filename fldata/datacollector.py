import torch
from typing import Any, List
import numpy as np
from abc import ABC, abstractmethod
from fldata.dataset import get_dataset

class AbstractCollector(ABC):
    @abstractmethod
    def set_centralized_dataset(self, name: str):
        '''Prepare centralized dataset.'''
        pass

    @abstractmethod
    def extract_data(self):
        '''Extract numpy data from centralized dataset.'''
        pass

    @abstractmethod
    def check_and_adjust_dimension(self):
        '''Adjust dimension of extracted data to be given configuration.'''
        pass 

    @abstractmethod
    def get_data(self):
        '''Return own data.'''
        pass


class BaseDataCollector(AbstractCollector):
    def __init__(self, name: str, root: str):
        self.set_centralized_dataset(name, root)
        self.extract_data()
        self.check_and_adjust_dimension()

    def set_centralized_dataset(self, name: str, root: str):
        self.train_centre_dataset = get_dataset(name)(root, train=True)
        self.valid_centre_dataset = get_dataset(name)(root, train=True)

    def extract_data(self):
        raise NotImplementedError()

    def check_and_adjust_dimension(self):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()


class ImageDataCollector(BaseDataCollector):
    def __init__(self, name:str, root: str):
        super().__init__(name=name, root=root)

    def extract_data(self):
        self.train_images = convert_to_numpy(self.train_centre_dataset.data)
        self.train_labels = convert_to_numpy(self.train_centre_dataset.targets)
        self.valid_images = convert_to_numpy(self.valid_centre_dataset.data)
        self.valid_labels = convert_to_numpy(self.valid_centre_dataset.targets)

    def check_and_adjust_dimension(self):
        # Expected dimension for images is (N x H x W x C)
        self.train_images = adjust_dimension(self.train_images, 4)
        self.valid_images = adjust_dimension(self.valid_images, 4)
        # Expected dimension for labels is (N X CLS)
        self.train_labels = adjust_dimension(self.train_labels, 2)
        self.valid_labels = adjust_dimension(self.valid_labels, 2)
    
    def get_data(self):
        return (self.train_images, self.train_labels), (self.valid_images, self.valid_labels)

# =================
# Helper functions 
# =================

def adjust_dimension(array: np.ndarray, expected_dim: int):
    difference = expected_dim - array.ndim
    if difference >=0:
        array = add_dimension(array, difference)
    else:
        raise ValueError(f"Given data is {array.ndim}-D which is higher than expected {expected_dim}-D.")
    return array

def add_dimension(array: np.ndarray, increment: int) -> np.ndarray:
    """Add new axes to given numpy array.

    Args:
        array (np.ndarray): input data.
        increment (int): number of dimensions to increase.

    Returns:
        np.ndarray: The input array, but with additional number of axes.

    Examples:
        >>> x=np.array([[[0],[1],[2]]])
        >>> x.shape
        (1, 3, 1)
        >>> add_dimension(x, 2).shape
        (1, 3, 1, 1, 1)
    """
    for _ in range(increment):
        array = array[..., None]
    return array

def convert_to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, list):
        data = list_to_numpy(data)
    elif isinstance(data, torch.Tensor):
        data = list_to_numpy(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise NotImplementedError()

    return data


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch.Tensor to numpy.ndarray.

    Args:
        tensor (torch.Tensor): input data

    Returns:
        np.ndarray: numpy array with the same elements as input tensor.
    """
    return tensor.detach().cpu().numpy()


def list_to_numpy(li: List) -> np.ndarray:
    return np.array(li)


