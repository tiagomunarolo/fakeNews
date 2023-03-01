import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer

FIRST_RUN = False
FINAL_PATH = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/final_dataset.csv"
UNIFIED_DATASET = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/unified_dataset.csv"
G1_PATH = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/g1.csv"
AOS_FATOS_PATH = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/aos_fatos.csv"

if FIRST_RUN:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

ps = PorterStemmer()

REMOVE_DATA = ['verdade', 'falso', 'fake', 'fato', 'true']
REMOVE_WORDS = []


def get_synonyms():
    """

    :return:
    """
    synonyms = ['fake']
    for remove in REMOVE_DATA:
        words = wordnet.synsets(remove, lang="por")
        for word in words:
            for syn in word.lemmas(lang="por"):
                synonyms.append(syn.name())
    return set(synonyms)


def remove_stop_words(content):
    """

    :param content:
    :return:
    """
    review = re.sub(r'[^a-zA-Z]', ' ', content).lower().split()
    review = [x for x in review if x not in REMOVE_WORDS]
    review = [ps.stem(word) for word in review if word not in stopwords.words('portuguese')]
    review = ' '.join(review)
    return review


def create_final_dataset():
    """
    Build final dataset to be analyzed
    """
    global REMOVE_WORDS
    REMOVE_WORDS = get_synonyms()
    df = pd.read_csv(G1_PATH, index_col=0).reset_index(drop=True)
    df_drop = df['RESUMO'].apply(lambda x: "fato" in x.lower() and "fake" in x.lower())
    df['LABEL'] = df['RESUMO'].apply(lambda x: False if "#fake" in x.lower() else True)
    df = df[~df_drop]

    df2 = pd.read_csv(AOS_FATOS_PATH, index_col=0).reset_index(drop=True)
    df2.replace({"VERDADEIRO": True, "FALSO": False}, inplace=True)
    df2_temp = df2[~df2.LABEL]
    df2.drop_duplicates(subset=["RESUMO"], inplace=True)
    df2 = pd.concat([df2, df2_temp])
    df = pd.concat([df, df2])
    df.to_csv(path_or_buf=UNIFIED_DATASET, index_label=False)

    df['TEXT'] = df.apply(lambda x: f"{x['RESUMO']} {x['TEXTO']}", axis=1)
    df['TEXT'] = df['TEXT'].apply(remove_stop_words)
    df = df[['TEXT', 'LABEL']]
    df.to_csv(path_or_buf=FINAL_PATH, index_label=False)


# manage_dataset
create_final_dataset()