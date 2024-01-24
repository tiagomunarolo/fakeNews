"""
Define MODELS options
"""
from . import ROOT
import os

DATASET_PATH = ROOT + os.sep + "data" + os.sep
# PATH OF DATASETS
FINAL_PATH = f"{DATASET_PATH}/preprocessed.csv"
ORIGINAL_DATASET = f"{DATASET_PATH}/original.csv"
G1_PATH = f"{DATASET_PATH}/g1.csv"
FAKE_CORPUS = f"{DATASET_PATH}/fake_corpus.csv"
RUMOR_PATH = f"{DATASET_PATH}/rumor.csv"
GPT_PATH = f"{DATASET_PATH}/chatgpt.csv"
SENS_PATH = f"{DATASET_PATH}/sensacionalista.csv"
FOLHA_PATH = f"{DATASET_PATH}/folha_sp.csv"

RANDOM_FOREST = "RANDOM_FOREST"
DECISION_TREE = "DECISION_TREE"
LOGISTIC = "LOGISTIC"
XGBOOST = "XGBOOST"
BAYES = "BAYES"
LSTM = "LSTM"
SVM = "SVM"
CNN = "CNN"

MODELS = [
    LOGISTIC,
    XGBOOST,
    RANDOM_FOREST,
    SVM,
    DECISION_TREE,
    LSTM,
    CNN,
    BAYES
]