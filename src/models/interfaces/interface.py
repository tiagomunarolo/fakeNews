import os
import pickle
from dataclasses import dataclass
from src.logger import get_logger
from typing import Protocol

logger = get_logger(logger_name=__file__)


class Store(Protocol):
    path: str

    def path_exists(self) -> bool: ...

    def store_model(self, obj: any) -> None: ...

    def read_model(self): ...

    def set_path(self, path: str) -> None: ...


@dataclass
class ObjectStore(Store):
    """
    Generic Object Store Class
    """
    path: str = ""

    def set_path(self, path: str) -> None:
        self.path = path

    @property
    def path_exists(self) -> bool:
        """
        Check if path exists
        :return:
        """
        return os.path.exists(self.path)

    def store_model(self, obj) -> None:
        """
        Save models to ./models dir
        """
        with open(self.path, 'wb') as file:
            logger.info(msg=f"STORING_MODEL: {self.path}")
            pickle.dump(obj=obj, file=file)
            logger.info(msg=f"MODEL_STORED: {self.path}")

    def read_model(self):
        """
        Reads Stored model from provided dir
        superclass
        """
        logger.info(msg=f"READING_MODEL: {self.path}")
        if not self.path_exists:
            raise FileNotFoundError(f'{self.path} does not exists')
        with open(self.path, 'rb') as file:
            model = pickle.load(file=file)
            logger.info(msg=f"MODEL_LOADED: {self.path} COMPLETED")
            return model
