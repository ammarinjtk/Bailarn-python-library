"""
Global Constant
"""
import os


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# vector dimension
VECTOR_DIM = 300
SEQUENCE_LENGTH = 60

# Tag
PAD_TAG_INDEX = 0
TAG_START_INDEX = 1

# BEST
TAG_LIST = ["NN", "NR", "PPER", "PINT", "PDEM", "DPER", "DINT", "DDEM", "PDT", "REFX", "VV", "VA", "AUX", "JJA", "JJV", "ADV",
            "NEG", "PAR", "CL", "CD", "OD", "FXN", "FXG", "FXAV", "FXAJ", "COMP", "CNJ", "P", "IJ", "PU", "FWN", "FWV", "FWA", "FWX"]

TAG_INDEXER = {
    '<PAD>': 0,
    'ADV': 16,
    'AUX': 13,
    'CD': 20,
    'CL': 19,
    'CNJ': 27,
    'COMP': 26,
    'DDEM': 8,
    'DINT': 7,
    'DPER': 6,
    'FWA': 33,
    'FWN': 31,
    'FWV': 32,
    'FWX': 34,
    'FXAJ': 25,
    'FXAV': 24,
    'FXG': 23,
    'FXN': 22,
    'IJ': 29,
    'JJA': 14,
    'JJV': 15,
    'NEG': 17,
    'NN': 1,
    'NR': 2,
    'OD': 21,
    'P': 28,
    'PAR': 18,
    'PDEM': 5,
    'PDT': 9,
    'PINT': 4,
    'PPER': 3,
    'PU': 30,
    'REFX': 10,
    'VA': 12,
    'VV': 11
}

NUM_TAGS = len(TAG_LIST) + 1


DEFULT_MODEL_PATH = os.path.join(
    BASE_PATH, './models/dufualt_model_bi_lstm_crf.hdf5')
DEFAULT_WORD_INDEX_PATH = os.path.join(BASE_PATH, './pos_word_index.json')

NUM_WORD = 55710
