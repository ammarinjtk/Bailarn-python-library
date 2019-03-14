"""
Global Constant
"""

import string

EMBEDDING_SIZE = 300
WORD_INDEXER_SIZE = 54943
SEQUENCE_LENGTH = 60
DROPOUT_PROB = 0.5

NUM_FILTER = (32, 32, 32)
FILTER_WIDTH = (5, 3, 2)

HIDDEN_DIM = 50

# Spacebar
SPACEBAR = " "

# Escape Character
ESCAPE_WORD_DELIMITER = "\t"

# Tag
PAD_TAG_INDEX = 0

# TAG_LIST = ['POS','NEU','NEG']
TAG_LIST = ['POS', 'NEG']

TAG_START_INDEX = 0


NUM_TAGS = len(TAG_LIST)

DEFAULT_MODEL_PATH = "models/cnn_multichannel.h5"
DEFAULT_WORD_INDEX_PATH = 'sentiment_word_index.json'

# Random Seed
SEED = 1395096092
