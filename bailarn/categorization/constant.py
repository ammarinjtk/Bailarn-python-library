import os

# Text processing keras
WORD_INDEXER_SIZE = 158499
EMBEDDING_SIZE = 300
SEQUENCE_LENGTH = 200

# Handle unknown word
UNKNOWN_WORD = 'UNK'

# Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = BASE_DIR + '/models/cnn_xmtc_model.h5'

DEFAULT_THRESHOLD_SELECTION_PATH = os.path.join(BASE_DIR, 'threshold_selection.json')
DEFAULT_WORD_INDEX_PATH = os.path.join(BASE_DIR, 'word_index.json')
    
TAG_START_INDEX = 0
TAG_LIST = [
     'Mobile Operator',
     'AIS',
     'Nokia',
     'โทรศัพท์มือถือ',
     'แท็บเล็ต',
     'Apple',
     'Mobile OS',
     'ร้องทุกข์',
     'Gadget',
     '3G',
     'บริษัทไอที',
     'HTC Smartphone',
     'การเงิน',
     'สมาร์ทโฟน',
     'dtac',
     'Samsung',
     '4G',
     'Sony Smartphone',
     'i-mobile',
     'OPPO Smartphone',
     'งานแสดงและจำหน่ายสินค้าไอที',
     'Social Network',
     'Huawei Smartphone',
     'vivo Smartphone',
     'iPhone',
     'Asus Mobile',
     'คอมพิวเตอร์',
     'Xiaomi Smartphone',
     'ย้ายค่ายเบอร์เดิม',
     'โทรศัพท์',
     'Mobile Application',
     'iPad',
     'LG Smartphone',
     'BlackBerry',
     'truemove',
     'พูดคุย',
     'Lenovo Smartphone',
     'My By CAT']
