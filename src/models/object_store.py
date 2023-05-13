from dataclasses import dataclass
import os
import pickle


@dataclass
class ObjectStore:
    """
    Generic Object Store Class
    """

    def __init__(self, path):
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
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def read_model(self):
        """
        Reads Stored model from provided dir
        superclass
        """
        if not self.path_exists:
            raise FileNotFoundError(f'{self.path} does not exists')
        with open(self.path, 'rb') as file:
            model = pickle.load(file=file)
            return model
