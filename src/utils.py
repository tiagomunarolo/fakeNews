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
from typing import Tuple, List

import re
import nltk
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from crawler import DATASET_PATH
from src.logger import get_logger

FINAL_PATH = f"{DATASET_PATH}/preprocessed.csv"
ORIGINAL_DATASET = f"{DATASET_PATH}/original_dataset.csv"

# PATH OF DATASETS
G1_PATH = f"{DATASET_PATH}/g1.csv"
FAKE_CORPUS = f"{DATASET_PATH}/fake_corpus.csv"
RUMOR_PATH = f"{DATASET_PATH}/rumor.csv"
GPT_PATH = f"{DATASET_PATH}/chatgpt.csv"

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stemmer = SnowballStemmer(language="portuguese")
STOP_WORDS = stopwords.words('portuguese')
TRUE_WORDS = ['verdade', 'fato', 'real', 'veridico', 'exato', 'checar', 'verificar']
FALSE_WORDS = ['falsear', 'fake', 'mentir', 'fraudar', 'inverdade',
               'fingir', 'enganar', 'ocultar', 'inventar']
REMOVE_DATA = TRUE_WORDS + FALSE_WORDS

logger = get_logger(__file__)


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


def get_synonyms(words_to_check: list):
    """
    Returns list of synonyms based on provided words.
    This step ensures and avoid bias due to specific words
    :return:
    """
    synonyms = ['fake']
    for remove in words_to_check:
        words = wordnet.synsets(remove, lang="por")
        for word in words:
            for syn in word.lemmas(lang="por"):
                synonyms.append(str(syn.name()).lower())
    return set(synonyms)


def remove_stop_words(content, remove_words):
    """
    Remove all stop words for brazilian-portuguese
    :param remove_words: list - words list to be removed
    :param content: str - text of news
    :return:
    """
    # remove special characters
    txt = re.sub(r'http://\S+|https://\S+', ' ', content)  # Remove URLs
    txt = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    txt = re.sub(r'[^\w+]', ' ', txt).split()
    txt = [stemmer.stem(w) for w in txt if w not in remove_words and not w.isnumeric()]
    txt = [x for x in txt if len(x) > 1]  # Remove residuals
    txt = ' '.join(txt)
    return txt


def manage_input(text: str) -> List[str]:
    """
    Get text and remove stopwords + data cleansing.
    :param text: str :: input text
    :return:
    """
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)
    text = remove_stop_words(content=text, remove_words=remove_words)
    return [text]


def create_final_dataset() -> None:
    """
    Build final data to be analyzed
    """
    logger.info("Creating Final Dataset")
    # Get a list of words to be removed (avoid bias)
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)
    # columns_used
    columns_used = ['TEXT', 'TEXT_SIZE', 'LABEL', 'SOURCE']
    """STEP 1: G1 data"""
    # Reads g1.csv (G1 Fato ou Fake source)
    df_g1 = pd.read_csv(G1_PATH, index_col=0).reset_index(drop=True)
    # Drop entries that can be dubious (either false or true)
    df_keep = df_g1['RESUMO'].apply(lambda x: x.lower().startswith("é #fato") or x.lower().startswith("é #fake"))
    df_g1 = df_g1[df_keep]
    # Label data according to it's content
    df_g1_aux = pd.DataFrame()
    df_g1_aux2 = pd.DataFrame()
    df_g1_aux['TEXT'] = df_g1['RESUMO']
    df_g1_aux['LABEL'] = df_g1['RESUMO'].apply(lambda x: False if "#fake" in x.lower() else True)
    df_g1_aux2['TEXT'] = df_g1['TEXTO']
    df_g1_aux2['LABEL'] = True
    df_g1 = pd.concat([df_g1_aux, df_g1_aux2])
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
    remove_words = STOP_WORDS + list(remove_words)
    for ix, data in final_df.iterrows():
        text = data['TEXT']
        final_df.at[ix, 'TEXT'] = remove_stop_words(content=text, remove_words=remove_words)
        if int(ix) and int(ix) % 500 == 0:
            logger.info(f"FINAL_DATASET: "
                        f"{round(int(ix) / len(final_df), 2)} processed")
    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(path_or_buf=FINAL_PATH, index_label=False)


__all__ = [
    'create_final_dataset',
    'manage_input',
    'get_xy_from_dataset'
]
