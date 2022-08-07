import torch
import pytest
import numpy as np
from fldata.datacollector import adjust_dimension, tensor_to_numpy, add_dimension
from fldata.datacollector import ImageDataCollector

class TestImageDataCollector:
    def test_collected_mnist_image_shape(self):
        collector = ImageDataCollector(name='mnist', root='/data/')
        images, _ = collector.get_data()[0]
        assert images[0].shape == (28, 28, 1)

    def test_collected_mnist_label_shape(self):
        collector = ImageDataCollector(name='mnist', root='/data/')
        _, labels = collector.get_data()[0]
        assert labels[0].shape == (1,)
    
    def test_collected_cifar10_image_shape(self):
        collector = ImageDataCollector(name='cifar10', root='/data/')
        images, _ = collector.get_data()[0]
        assert images[0].shape == (32, 32, 3)

    def test_collected_cifar10_label_shape(self):
        collector = ImageDataCollector(name='cifar10', root='/data/')
        _, labels = collector.get_data()[0]
        assert labels[0].shape == (1,)


def test_tensor_to_numpy():
    tensor = torch.ones((2, 2))
    array = tensor_to_numpy(tensor)
    assert isinstance(array, np.ndarray)

def test_add_dimension():
    array = np.ones((2,2))
    array = add_dimension(array, increment=3)
    assert array.ndim == 5

def test_adjust_dimension_higher_than_expect():
    with pytest.raises(ValueError):
        array = np.ones((2,1,1,1))
        expected_dim = 3
        array = adjust_dimension(array, expected_dim)

def test_adjust_dimension_lower_than_expect():
    array = np.ones((2,1,1))
    expected_dim = 4
    array = adjust_dimension(array, expected_dim)
    assert array.ndim == expected_dim

def test_adjust_dimension_same_as_expect():
    array = np.ones((2,1))
    expected_dim = 2
    array = adjust_dimension(array, expected_dim)
    assert array.ndim == expected_dim

