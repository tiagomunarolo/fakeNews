"""
This file manages all provided datasets

Generated datasets:
-Aos fatos: extracted with bs4 + selenium (g1.py)
-G1 Fato ou Fake: extracted with bs4 + selenium (aos_fatos.py)

Used datasets:
- rumor_elections_2018.csv
    source: (https://www.kaggle.com/datasets/caiovms/brazilian-election-fake-news-2018?select=
    rumor.csv)

- fake_corpus.csv
    source: https://github.com/roneysco/Fake.br-Corpus/tree/master/full_texts
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from scrapper import DATA_PATH

FIRST_RUN = False

FINAL_PATH = f"{DATA_PATH}/final_dataset.csv"
UNIFIED_DATASET = f"{DATA_PATH}/unified_dataset.csv"

# PATH OF DATASETS
G1_PATH = f"{DATA_PATH}/g1.csv"
AOS_FATOS_PATH = f"{DATA_PATH}/aos_fatos.csv"
FAKE_CORPUS = f"{DATA_PATH}/fake_corpus.csv"
RUMOR_PATH = f"{DATA_PATH}/rumor.csv"
GPT_PATH = f"{DATA_PATH}/chatgpt.csv"

if FIRST_RUN:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

stemmer = SnowballStemmer(language="portuguese")
STOP_WORDS = stopwords.words('portuguese')
REMOVE_DATA = ['verdade', 'falso', 'fake',
               'fato', 'true', 'mentira',
               'verificar', 'checar']


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
                synonyms.append(syn.name())
    return set(synonyms)


def remove_stop_words(content, remove_words):
    """
    Remove all stop words for brazilian-portuguese
    :param remove_words: list - words list to be removed
    :param content: str - text of news
    :return:
    """
    # remove special characters
    txt = re.sub('http://\S+|https://\S+', ' ', content)  # Remove URLs
    txt = normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub(r'[^\w+]', ' ', txt).split()
    txt = [stemmer.stem(w) for w in txt if w not in remove_words and not w.isnumeric()]
    txt = [x for x in txt if len(x) > 1]  # Remove residuals
    txt = ' '.join(txt)
    return txt


def generate_dataset_for_input(text):
    """
    Get preprocessed text
    :param text: str
    :return:
    """
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)
    df = pd.DataFrame(data={"TEXT": [text]})
    df['TEXT'] = df['TEXT'].apply(lambda x: remove_stop_words(content=x, remove_words=remove_words))
    return df.TEXT


def create_final_dataset():
    """
    Build final dataset to be analyzed
    """
    # Get a list of words to be removed (avoid bias)
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)

    """STEP 1: G1 dataset"""
    # Reads g1.csv (G1 Fato ou Fake source)
    df = pd.read_csv(G1_PATH, index_col=0).reset_index(drop=True)
    # Drop entries that can be dubious (either false or true)
    df_drop = df['RESUMO'].apply(lambda x: "fato" in x.lower() and "fake" in x.lower())
    df = df[~df_drop]
    # Label data according to it's content
    df['LABEL'] = df['RESUMO'].apply(lambda x: False if "#FAKE" in x else True)
    df['TEXT'] = df.apply(lambda x: f"{x['RESUMO']} {x['TEXTO']}", axis=1)
    df = df[['TEXT', 'LABEL']]

    """STEP 2: Aos fatos dataset"""
    # Reads aos_fatos.csv (Aos fatos data source)
    df2 = pd.read_csv(AOS_FATOS_PATH, index_col=0).reset_index(drop=True)
    df2.rename(columns={"TEXTO": "TEXT"}, inplace=True)
    df2 = df2[['TEXT', 'LABEL']]

    """STEP 3: Fake corpus dataset"""
    df3 = pd.read_csv(FAKE_CORPUS, index_col=0).reset_index(drop=True)
    df3.rename(columns={"label": "LABEL", "preprocessed_news": "TEXT"}, inplace=True)
    df3.LABEL = df3.LABEL.replace({"fake": False, "true": True})

    """STEP 4: Kaggle rumor election dataset"""
    df4 = pd.read_csv(RUMOR_PATH, index_col=0, sep=";").reset_index(drop=True)
    df4.rename(columns={"texto": "TEXT", "rotulo": "LABEL"}, inplace=True)
    df4.LABEL = df4.LABEL.replace({"FALSO": False, "VERDADE": True})
    df4 = df4[['TEXT', 'LABEL']]

    """STEP 5: Chat GPT Fake data set"""
    df5 = pd.read_csv(GPT_PATH, index_col=0).reset_index(drop=True)
    df5 = df5[['TEXT', 'LABEL']]

    # concat dataframes
    final_df = pd.concat([df, df2, df3, df4, df5])
    final_df.TEXT = final_df.TEXT.str.lower()
    final_df.reset_index(inplace=True, drop=True)
    final_df.to_csv(path_or_buf=UNIFIED_DATASET)

    remove_words = STOP_WORDS + list(remove_words)
    final_df['TEXT'] = final_df['TEXT'].apply(
        lambda x: remove_stop_words(content=x, remove_words=remove_words))
    final_df.drop_duplicates(inplace=True)
    # Remove texts with text size >=500 or lower than 15
    final_df = final_df[~final_df.TEXT.apply(lambda x: len(x.split()) <= 15 or len(x.split()) > 500)]
    final_df.to_csv(path_or_buf=FINAL_PATH)