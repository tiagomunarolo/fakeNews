"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from src.logger.logging import get_logger
from src.models.interfaces import Store, ParameterSciKit
from src.models.object_store import ObjectStore

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

logger = get_logger(__file__)


class TfClassifier:
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, parameters: ParameterSciKit, store: Store = ObjectStore()):
        """Init Model"""
        self.store = store
        self.store.set_path(path=f"./{parameters.model_name}.model")
        self.model_type = parameters.model_type
        self.param_grid = parameters.param_grid
        self.model_name = parameters.model_name
        self.model = None
        self.tf_vector = None

    def fit(self, X: any, y: any, refit: bool = False) -> None:
        """
        Fit Generic provided models with GridSearchCV
        :param y: Array like, Output
        :param X: Array like, Input
        :type refit: bool: Force fit models if it no longer exists
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
            return

        estimator = self.model_type(random_state=42)
        pipeline = Pipeline([
            # Ignore terms that appears less than 10 and more than 1000 docs
            # Remove infrequent and too frequent words
            ('tfidf', TfidfVectorizer(min_df=10, max_df=1000)),
            # Reduces Input Dimension via Chi_square Selection
            ('k_best', SelectKBest(score_func=chi2, k=5000)),
            (f'{self.model_name}', estimator)
        ])

        tf_idf_params = {
            # Consider UniGrams and BiGrams
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__norm': ['l1', 'l2']
        }

        param_grid = {f"{self.model_name}__{k}": v for k, v in self.param_grid.items()}
        param_grid = {**param_grid, **tf_idf_params}
        logger.info(msg=f"FITTING_MODEL: {self.model_name} STARTED")
        grid = GridSearchCV(estimator=pipeline,
                            param_grid=param_grid,
                            cv=5,
                            verbose=5,
                            scoring=('r2', 'roc_auc', 'f1'),
                            refit='f1',
                            )

        grid.fit(X=X, y=y.astype(int))
        logger.info(msg=f"MODEL_FITTING: {self.model_name} DONE!")
        # select best models
        self.model = grid.best_estimator_
        logger.info(msg=f"TRAINING_SCORES :: {self.model_name} :: {grid.best_score_}")
        # Store models
        self.store.store_model(obj=self)

    def predict(self, X):
        """
        :parameter: X: Text list to be predicted
        Generates prediction, given a text
        """
        if not self.model:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__

        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)


__all__ = ['TfClassifier', ]
