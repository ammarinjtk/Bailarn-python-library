import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATH = os.path.join(BASE_PATH,'./models/phone_lower')

VECTOR_DIM = 100
N_EXPAND = 10
CBOW_SG_FLAG = 1
WINDOW=5
MIN_COUNT=5
WORKERS=4