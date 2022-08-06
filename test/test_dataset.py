import pytest
import numpy as np
from fldata.dataset import get_dataset
from fldata.dataset import load_data


class TestLoadDataMNIST:
    def test_load_data_return_type():
        images, labels = load_data(name='centre_mnist')
        assert isinstance(images, np.ndarray) \
            and isinstance(labels, np.ndarray) 
    
    def test_load_data_shape():
        images, labels = load_data(name='centre_mnist')
        assert images[0].shape == (28,28,1) and labels[0].shape == (1,)

    def test_load_data_dim():
        images, labels = load_data(name='centre_mnist')
        assert images.ndim==4 and labels.ndim==2

def test_raiseError_for_nonexisting_dataset():
    with pytest.raises(NameError):
        _ = get_dataset(name="wired_dataset")

