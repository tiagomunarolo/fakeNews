"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import warnings

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from src.logger.logging import get_logger
from src.preprocess.clean_transformer import CleanTextTransformer
from src.models.interfaces import ParameterSciKit
from src.models.object_store import ObjectStore

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

logger = get_logger(__file__)


class TfClassifier:
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, parameters: ParameterSciKit):
        """Init Model"""
        self.store = ObjectStore(path=f"./{parameters.model_name}.model")
        self.model_type = parameters.model_type
        self.param_grid = parameters.param_grid
        self.model_name = parameters.model_name
        self.model = None

    def fit(self, X: any, y: any, refit: bool = False, clean_data=False) -> None:
        """
        Fit Generic provided models with GridSearchCV
        :param y: Array like, Output
        :param X: Array like, Input
        :param clean_data: Force Clean Data

        :type refit: bool: Force fit models if it no longer exists

        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
            return

        if clean_data:
            X = CleanTextTransformer().fit_transform(X)

        y = y.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, random_state=42, stratify=y, test_size=0.1)

        folds = StratifiedKFold(n_splits=5).split(X_train, y_train)
        estimator = self.model_type(random_state=42)
        pipeline = Pipeline([
            # Ignore terms that appears less than 10 and more than 10000 docs
            # Remove infrequent and too frequent words
            ('tfidf', TfidfVectorizer(min_df=10, max_df=10000,
                                      max_features=100000)),
            # Reduces Input Dimension via Chi_square Selection
            ('k_best', SelectKBest(score_func=chi2, k=10000)),
            (f'{self.model_name}', estimator)
        ])

        tf_idf_params = {
            # Consider Uni, Bi, Trigrams --> showed best results
            'tfidf__ngram_range': [(1, 2)],
            # norm = l1 -> showed better results than l2
            'tfidf__norm': ['l1']
        }

        param_grid = {f"{self.model_name}__{k}": v for k, v in self.param_grid.items()}
        param_grid = {**param_grid, **tf_idf_params}
        logger.info(msg=f"{self.model_name} : FIT STARTED")
        self.model = GridSearchCV(estimator=pipeline,
                                  param_grid=param_grid,
                                  cv=folds,
                                  verbose=5,
                                  scoring=('r2', 'roc_auc', 'f1'),
                                  refit='f1',
                                  )

        self.model.fit(X=X_train, y=y_train)
        logger.info(msg=f"{self.model_name} : FIT DONE")
        # select best models
        logger.info(msg=f"{self.model_name} : BEST_ESTIMATOR :: {self.model.best_estimator_}")
        logger.info(msg=f"{self.model_name} : TRAINING_SCORES :: {self.model.best_score_}")
        # Store models / drop generator
        del self.model.cv
        self.store.store_model(obj=self)
        logger.info(f"{self.model_name} : SCORE_TEST => {self.model.score(X_test, y_test, )}")

    def predict(self, X, clean_data=True):
        """
        :parameter: X: Text list to be predicted
        :param clean_data: Force Clean Data
        Generates prediction, given a text
        """
        if not self.model:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
        if clean_data:
            X = CleanTextTransformer().fit_transform(X=X)
        return self.model.predict(X=X)


__all__ = ['TfClassifier', ]
