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


class Store(Protocol):
    path: str

    def path_exists(self) -> bool: ...

    def store_model(self, obj: any) -> None: ...

    def read_model(self): ...

    def set_path(self, path: str) -> None: ...
