import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer
from unicodedata import normalize
from nltk.corpus import stopwords
import re
import nltk
import spacy


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
        self._load_libs()
        # Get stop words
        self.stop = " ".join(stopwords.words('portuguese'))
        self.stop = normalize('NFKD', self.stop).encode('ASCII', 'ignore').decode('ASCII').split()
        self.remove = ['verdade', 'fato', 'real', 'fake', 'mentir', 'falso']
        self.nlp = spacy.load("pt_core_news_lg")
        stemmer = SnowballStemmer(language="portuguese")
        self.remove = [stemmer.stem(w) for w in self.remove]

    @staticmethod
    def _load_libs() -> None:
        """
        Load required NLTK libs
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        spacy.load('pt_core_news_lg')

    def clean_text(self, content: str) -> str:
        """
        Remove all stop words for brazilian-portuguese
        :param content: str - text of news
        :return:
        """

        def check_word(word):
            if len(word) <= 2:
                return False
            if word.isnumeric():
                return False
            if word in self.stop:
                return False
            if self.nlp.vocab[str(word)].is_stop:
                return False
            if word.startswith(tuple(self.remove)):
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
        return X.apply(self.clean_text)
