"""
Utils file
"""
import os
import pandas as pd
from typing import Tuple


def get_xy_from_dataset(path: str = "") \
        -> Tuple[pd.Series, pd.Series]:
    """
    Reads Training Dataset
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found!')

    data = pd.read_csv(path, index_col=0)
    X = data.TEXT
    Y = data.LABEL
    return X, Y
