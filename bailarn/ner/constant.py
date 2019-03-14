"""
Global Constant
"""

import string


EMBEDDING_SIZE = 128
WORD_INDEXER_SIZE = 55709

SEQUENCE_LENGTH = 2000
# Spacebar
SPACEBAR = " "

# Escape Character
ESCAPE_WORD_DELIMITER = "\t"
ESCAPE_TAG_DELIMITER = "\v"

# data dimention
DATA_DIM = 300

# Tag
# PAD_TAG_INDEX = 0
# NON_SEGMENT_TAG_INDEX = 1
TAG_START_INDEX = 0
TAG_LIST = ['X', 'DTM_B', 'DTM_I',
            'DES_B', 'DES_I',
            'TTL_B', 'TTL_I',
            'BRN_B', 'BRN_I',
            'PER_B', 'PER_I',
            'MEA_B', 'MEA_I',
            'NUM_B', 'NUM_I',
            'LOC_B', 'LOC_I',
            'TRM_B', 'TRM_I',
            'ORG_B', 'ORG_I',
            'ABB_ORG_B', 'ABB_ORG_I',
            'ABB_LOC_B', 'ABB_LOC_I',
            'ABB_DES_B', 'ABB_DES_I',
            'ABB_PER_B', 'ABB_PER_I',
            'ABB_TTL_B', 'ABB_TTL_I',
            'ABB_B', 'ABB_I',
            'ABB', 'DDEM',
            'NAME_B', '__',
            'O', 'UNK']

NUM_TAGS = len(TAG_LIST)

DEFAULT_MODEL_PATH = "./models/ner_crf_e40.h5"
DEFAULT_WORD_INDEX_PATH = './ner_word_index.json'

# Random Seed
SEED = 1395096092
