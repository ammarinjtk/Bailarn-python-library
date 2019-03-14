import os

# Random Seed
SEED = 1395096092

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFULT_MODEL_PATH = BASE_DIR + "/models/w2v.bin"

# Unknown word
UNKNOWN_WORD = 'UNK'

EMBEDDING_SIZE = 300

# sg=1 means to use skip-gram technique
CBOW_SG_FLAG = 1
