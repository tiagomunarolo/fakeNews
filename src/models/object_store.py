from src.models.interfaces import Store
from dataclasses import dataclass
import os
import pickle
from src.logger.logging import get_logger

logger = get_logger(__file__)


@dataclass
class ObjectStore(Store):
    """\
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
