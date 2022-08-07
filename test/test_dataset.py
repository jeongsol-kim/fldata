import pytest
from fldata.dataset import get_dataset


def test_raiseError_for_nonexisting_dataset():
    with pytest.raises(NameError):
        _ = get_dataset(name="wired_dataset")
