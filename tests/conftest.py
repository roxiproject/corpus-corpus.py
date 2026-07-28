import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
