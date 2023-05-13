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
import pandas as pd
from typing import Tuple
from src.logger import get_logger
from src.constants import *

logger = get_logger(__file__)


def load_dataset(path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Reads Training Dataset
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found!')

    data = pd.read_csv(path, index_col=0)
    X = data.TEXT
    Y = data.LABEL
    return X, Y


def build_dataset() -> None:
    """
    Builds dataset to be analyzed / trained
    """
    from .transformer import CustomTransformer
    logger.info("Creating Final Dataset")
    # columns_used
    columns_used = ['TEXT', 'TEXT_SIZE', 'LABEL', 'SOURCE']
    """STEP 1: G1 data"""
    # Reads g1.csv (G1 Fato ou Fake source)
    df_g1 = pd.read_csv(G1_PATH, index_col=0).reset_index(drop=True)
    df_g1['TEXT'] = df_g1['RESUMO'] + " " + df_g1['TEXTO']
    df_g1.TEXT = df_g1.TEXT.str.lower()
    # keep true labels
    df_g1_true = df_g1[df_g1.LABEL.isin([True])]
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
    final_df.TEXT = CustomTransformer().fit_transform(X=final_df.TEXT)
    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(path_or_buf=FINAL_PATH, index_label=False)
