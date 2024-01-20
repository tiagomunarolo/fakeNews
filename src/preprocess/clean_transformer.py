import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer
from unicodedata import normalize
from nltk.corpus import stopwords
import re
import nltk
import spacy
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


class CleanTextTransformer(BaseEstimator, TransformerMixin):

    def fit(self, *args, **kwargs):
        """
        Fit Model
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        return self

    def __init__(self):
        self.nlp = self._load_libs()
        # Spacy Stop Words
        _to_remove = ' '.join(self.nlp.Defaults.stop_words)
        _to_remove = normalize('NFKD', _to_remove). \
            encode('ASCII', 'ignore'). \
            decode('ASCII').split()
        # Nltk stop words
        _to_remove_nltk = " ".join(stopwords.words('portuguese'))
        _to_remove_nltk = normalize('NFKD', _to_remove_nltk). \
            encode('ASCII', 'ignore'). \
            decode('ASCII').split()
        # Word Stemming for common words
        common_words = ['verdade', 'fato', 'real', 'fake', 'mentir', 'falso']
        common_words = [SnowballStemmer(language="portuguese").stem(w) for w in common_words]
        self.common_words = common_words
        # Words to be removed
        self.remove = set(_to_remove + _to_remove_nltk + common_words)

    @staticmethod
    def _load_libs() -> spacy.language.Language:
        """
        Load required NLTK libs
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        try:
            return spacy.load('pt_core_news_lg')
        except OSError:
            spacy.cli.download("pt_core_news_lg")
            return spacy.load('pt_core_news_lg')

    def clean_text(self, content: str) -> str:
        """
        Remove all stop words for brazilian-portuguese
        :param content: str - text of news
        :return:
        """

        def check_word(word):
            if word in self.remove:
                return False
            if len(word) <= 2:
                return False
            if word.startswith(tuple(self.common_words)):
                return False
            return True

        txt = re.sub(r'http://\S+|https://\S+', ' ', content)  # Remove URLs
        txt = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
        txt = re.sub(r'[^a-zA-Z]', ' ', txt)  # keep only characters
        txt = ' '.join([x.lemma_ for x in self.nlp(txt)]).lower().split()
        txt = [w for w in txt if check_word(w)]
        txt = ' '.join(txt)
        return txt.lower()

    def transform(self, X, y=None):
        """
        Transform and Clean Data
        Parameters
        ----------
        X => Data to be clean
        y => ignored

        Returns
        -------

        """
        if isinstance(X, str):
            X = pd.Series(X)
        elif isinstance(X, list):
            X = pd.Series(X)
        return X.parallel_apply(self.clean_text)
