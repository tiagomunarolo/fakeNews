from typing import Protocol


class ParameterLstm(Protocol):
    # HyperParameters of generic model
    max_features: int  # max words in data dictionary
    pad_len: int
    layer_1: int
    layer_2: int
    layer_3: int
    epochs: int
    batch_size: int
    # model metadata
    model_name: str


class ParameterCnn(Protocol):
    # HyperParameters of generic model
    vocab_size: int  # max words in data dictionary
    pad_len: int
    epochs: int
    batch_size: int
    transform_size: int
    # model metadata
    model_name: str


class ParameterSciKit(Protocol):
    # HyperParameters of generic model
    model_name: str
    model_type: any
    param_grid: dict
