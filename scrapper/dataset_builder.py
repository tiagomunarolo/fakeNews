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
from scrapper import DATASET_PATH

FINAL_PATH = f"{DATASET_PATH}/final_dataset.csv"
UNIFIED_DATASET = f"{DATASET_PATH}/unified_dataset.csv"

# PATH OF DATASETS
G1_PATH = f"{DATASET_PATH}/g1.csv"
AOS_FATOS_PATH = f"{DATASET_PATH}/aos_fatos.csv"
FAKE_CORPUS = f"{DATASET_PATH}/fake_corpus.csv"
RUMOR_PATH = f"{DATASET_PATH}/rumor.csv"
GPT_PATH = f"{DATASET_PATH}/chatgpt.csv"

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = SnowballStemmer(language="portuguese")
STOP_WORDS = stopwords.words('portuguese')
TRUE_WORDS = ['verdade', 'fato', 'real', 'veridico', 'exato', 'checar', 'verificar']
FALSE_WORDS = ['falsear', 'fake', 'mentir', 'fraudar', 'inverdade',
               'fingir', 'enganar', 'ocultar', 'inventar']
REMOVE_DATA = TRUE_WORDS + FALSE_WORDS


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


def generate_dataset_for_input(df: pd.DataFrame):
    """
    Get text and remove stopwords + data cleansing.
    :param df: pd.Dataframe
    :return:
    """
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)
    return df.TEXT.apply(lambda x: remove_stop_words(content=x, remove_words=remove_words))


def create_final_dataset():
    """
    Build final dataset to be analyzed
    """
    print("Creating Final Dataset")
    # Get a list of words to be removed (avoid bias)
    remove_words = get_synonyms(words_to_check=REMOVE_DATA)
    # columns_used
    columns_used = ['TEXT', 'TEXT_SIZE', 'LABEL', 'SOURCE']
    """STEP 1: G1 dataset"""
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
    print("G1 Dataset Done")
    """STEP 2: Aos fatos dataset"""
    # Reads aos_fatos.csv (Aos fatos data source)
    # df_aos_fatos = pd.read_csv(AOS_FATOS_PATH, index_col=0).reset_index(drop=True)
    # df_aos_fatos.rename(columns={"TEXTO": "TEXT"}, inplace=True)
    # df_aos_fatos['TEXT_SIZE'] = df_aos_fatos.TEXT.apply(lambda x: len(x.split()))
    # df_aos_fatos = df_aos_fatos[columns_used]

    """STEP 3: Fake corpus dataset"""
    df_corpus = pd.read_csv(FAKE_CORPUS, index_col=0).reset_index(drop=True)
    df_corpus.rename(columns={"label": "LABEL", "preprocessed_news": "TEXT"}, inplace=True)
    df_corpus.LABEL = df_corpus.LABEL.replace({"fake": False, "true": True})
    df_corpus['TEXT_SIZE'] = df_corpus.TEXT.apply(lambda x: len(x.split()))
    df_corpus['SOURCE'] = 'CORPUS_BR'
    df_corpus = df_corpus[columns_used]
    print("Fake.BR Corpus Dataset Done")

    """STEP 4: Kaggle rumor election dataset"""
    df_rumor = pd.read_csv(RUMOR_PATH, index_col=0, sep=";").reset_index(drop=True)
    df_rumor.rename(columns={"texto": "TEXT", "rotulo": "LABEL"}, inplace=True)
    df_rumor.LABEL = df_rumor.LABEL.replace({"FALSO": False, "VERDADE": True})
    df_rumor['TEXT_SIZE'] = df_rumor.TEXT.apply(lambda x: len(x.split()))
    df_rumor['SOURCE'] = 'KAGGLE_RUMOR'
    df_rumor = df_rumor[columns_used]
    print("Kaggle Rumor Dataset Done")

    """STEP 5: Chat GPT Fake data set"""
    df_gpt = pd.read_csv(GPT_PATH, index_col=0).reset_index(drop=True)
    df_gpt = df_gpt[['TEXT', 'LABEL']]
    df_gpt['TEXT_SIZE'] = df_gpt.TEXT.apply(lambda x: len(x.split()))
    df_gpt['SOURCE'] = 'CHAT_GPT'
    df_gpt = df_gpt[columns_used]
    print("ChatGPT Dataset Done")

    # concat dataframes
    final_df = pd.concat([df_g1, df_corpus, df_rumor, df_gpt])
    final_df.TEXT = final_df.TEXT.str.lower()
    final_df.reset_index(inplace=True, drop=True)
    final_df.to_csv(path_or_buf=UNIFIED_DATASET, index_label=False)
    print("Final Unified Dataset Done")
    remove_words = STOP_WORDS + list(remove_words)
    for ix, data in final_df.iterrows():
        text = data['TEXT']
        final_df.at[ix, 'TEXT'] = remove_stop_words(content=text, remove_words=remove_words)
        if int(ix) and int(ix) % 100 == 0:
            print(f"FINAL_DATASET: {ix}/{len(final_df)} processed")
    remove_lower_false = final_df[(~final_df.LABEL) & (final_df.TEXT_SIZE <= 15)]
    final_df = final_df[~final_df.index.isin(remove_lower_false.index)]
    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(path_or_buf=FINAL_PATH, index_label=False)
