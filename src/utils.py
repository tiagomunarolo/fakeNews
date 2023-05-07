"""
Utils file
This file manages all provided datasets

Generated datasets:
-G1 Fato ou Fake: extracted with bs4 + selenium

Used datasets:
- rumor_elections_2018.csv
    source: (https://www.kaggle.com/datasets/caiovms/brazilian-election-fake-news-2018?select=
    rumor.csv)

- fake_corpus.csv
    source: https://github.com/roneysco/Fake.br-Corpus/tree/master/full_texts
"""
import os
import pandas as pd
import swifter
from typing import Tuple, List

import re
import nltk
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.stem import SnowballStemmer
from crawler import DATASET_PATH
from src.logger import get_logger
import spacy

# PATH OF DATASETS
FINAL_PATH = f"{DATASET_PATH}/preprocessed.csv"
ORIGINAL_DATASET = f"{DATASET_PATH}/original.csv"
G1_PATH = f"{DATASET_PATH}/g1.csv"
FAKE_CORPUS = f"{DATASET_PATH}/fake_corpus.csv"
RUMOR_PATH = f"{DATASET_PATH}/rumor.csv"
GPT_PATH = f"{DATASET_PATH}/chatgpt.csv"

# NLTK dependencies
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = stopwords.words('portuguese')
TRUE_WORDS = ['verdade', 'fato', 'real']
FALSE_WORDS = ['fake', 'mentir', 'falso']
REMOVE_DATA = TRUE_WORDS + FALSE_WORDS
# logger
logger = get_logger(__file__)

# Spacy
spacy.load('pt_core_news_lg')
nlp = spacy.load("pt_core_news_lg")

# Stemmer
stemmer = SnowballStemmer(language="portuguese")
REMOVE_DATA = [stemmer.stem(w) for w in REMOVE_DATA]


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


def clean_text(content: str):
    """
    Remove all stop words for brazilian-portuguese
    :param content: str - text of news
    :return:
    """
    txt = re.sub(r'http://\S+|https://\S+', ' ', content)  # Remove URLs
    txt = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    txt = re.sub(r'[^\w+]', ' ', txt)
    txt = ' '.join([x.lemma_ for x in nlp(txt)]).lower().split()
    txt = [w for w in txt if not nlp.vocab[str(w)].is_stop and not w.isnumeric()]
    txt = [w for w in txt if w not in STOP_WORDS]
    txt = [w for w in txt if not w.startswith(tuple(REMOVE_DATA))]
    txt = [w for w in txt if len(w) > 2]  # Remove residuals
    txt = ' '.join(txt)
    return txt.lower()


def manage_input(text: str) -> List[str]:
    """
    Get text and remove stopwords + data cleansing.
    :param text: str :: input text
    :return:
    """
    text = clean_text(content=text)
    return [text]


def create_final_dataset() -> None:
    """
    Build final data to be analyzed
    """
    logger.info("Creating Final Dataset")
    # columns_used
    columns_used = ['TEXT', 'TEXT_SIZE', 'LABEL', 'SOURCE']
    """STEP 1: G1 data"""
    # Reads g1.csv (G1 Fato ou Fake source)
    df_g1 = pd.read_csv(G1_PATH, index_col=0).reset_index(drop=True)
    df_g1['TEXT'] = df_g1['RESUMO'] + " " + df_g1['TEXTO']
    df_g1.TEXT = df_g1.TEXT.str.lower()
    # keep true labels
    df_g1_true = df_g1[df_g1.LABEL == True]
    df_g1 = df_g1[~df_g1.index.isin(df_g1_true.index)]
    # Drop entries that can be dubious (either false or true)
    df_g1 = df_g1[df_g1['TEXT'].apply(lambda x: x.startswith("é #fato") or x.startswith("é #fake"))]
    # Label data according to it's content
    df_g1['LABEL'] = df_g1['TEXT'].apply(lambda x: False if "#fake" in x.lower() else True)
    df_g1 = pd.concat([df_g1_true, df_g1])
    df_g1['TEXT_SIZE'] = df_g1.TEXT.apply(lambda x: len(x.split()))
    df_g1['SOURCE'] = 'G1'
    df_g1 = df_g1[columns_used]
    logger.info("G1 Dataset Done")

    """STEP 2: Fake corpus data"""
    df_corpus = pd.read_csv(FAKE_CORPUS, index_col=0).reset_index(drop=True)
    df_corpus.rename(columns={"label": "LABEL", "preprocessed_news": "TEXT"}, inplace=True)
    df_corpus.LABEL = df_corpus.LABEL.replace({"fake": False, "true": True})
    df_corpus['TEXT_SIZE'] = df_corpus.TEXT.apply(lambda x: len(x.split()))
    df_corpus['SOURCE'] = 'CORPUS_BR'
    df_corpus = df_corpus[columns_used]
    logger.info("Fake.BR Corpus Dataset Done")

    """STEP 3: Kaggle rumor election data"""
    df_rumor = pd.read_csv(RUMOR_PATH, index_col=0, sep=";").reset_index(drop=True)
    df_rumor.rename(columns={"texto": "TEXT", "rotulo": "LABEL"}, inplace=True)
    df_rumor.LABEL = df_rumor.LABEL.replace({"FALSO": False, "VERDADE": True})
    df_rumor['TEXT_SIZE'] = df_rumor.TEXT.apply(lambda x: len(x.split()))
    df_rumor['SOURCE'] = 'KAGGLE_RUMOR'
    df_rumor = df_rumor[columns_used]
    logger.info("Kaggle Rumor Dataset Done")

    """STEP 4: Chat GPT Fake data set"""
    df_gpt = pd.read_csv(GPT_PATH, index_col=0).reset_index(drop=True)
    df_gpt = df_gpt[['TEXT', 'LABEL']]
    df_gpt['TEXT_SIZE'] = df_gpt.TEXT.apply(lambda x: len(x.split()))
    df_gpt['SOURCE'] = 'CHAT_GPT'
    df_gpt = df_gpt[columns_used]
    logger.info("ChatGPT Dataset Done")

    # concat dataframes
    final_df = pd.concat([df_g1, df_corpus, df_rumor, df_gpt])
    final_df.TEXT = final_df.TEXT.str.lower()
    final_df.reset_index(inplace=True, drop=True)
    final_df.to_csv(path_or_buf=ORIGINAL_DATASET, index_label=False)
    logger.info("Final Unified Dataset Done")
    final_df.TEXT = final_df.TEXT. \
        swifter.progress_bar(True). \
        apply(clean_text)
    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(path_or_buf=FINAL_PATH, index_label=False)


__all__ = [
    'create_final_dataset',
    'manage_input',
    'get_xy_from_dataset'
]
