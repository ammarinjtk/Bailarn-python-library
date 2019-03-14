from setuptools import setup, find_packages

setup(
    # Application name:
    name="bailarn",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Amarin Jettakul",
    author_email="ammarinjtk@gmail.com",

    # Packages
    packages=["bailarn", "bailarn.instant_model", "bailarn.utils", "bailarn.tokenizer",
              "bailarn.categorization", "bailarn.word_embedder", "bailarn.sentiment", "bailarn.pos", "bailarn.ner",
              "bailarn.keyword_expansion"],

    # Include additional files into the package
    include_package_data=True,

    # data_files=[
    #     ('bailarn/tokenizer/models', '0014-0.0443.hdf5'),
    #     ('bailarn/word_embedder/models',
    #      ['w2v.bin', 'w2v.bin.syn1neg.npy', 'w2v.bin.wv.syn0.npy']),
    #     ('bailarn/categorization/models', 'cnn_xmtc_model.h5'),
    #     ('bailarn/ner/models', 'ner_crf_e40.h5'),
    #     ('bailarn/pos/models', 'dufualt_model_bi_lstm_crf.hdf5'),
    #     ('bailarn/keyword_expansion/models',
    #      ['phone_lower', 'phone_lower.syn1neg.npy', 'phone_lower.wv.syn0.npy']),
    #     ('bailarn/sentiment/models', 'cnn_multi_2tag_e3.h5'),
    #     ('bailarn/categorization',
    #      ['categorization_word_index.json', 'threshold_selection.json']),
    #     ('bailarn/ner', 'ner_word_index.pickle'),
    #     ('bailarn/pos', 'pos_word_index.json'),
    #     ('bailarn/sentiment', 'sentiment_word_index.pickle'),
    # ],

    # # Details
    url="https://github.com/KawinL/Thai_NLP_platform",

    license="LICENSE",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    install_requires=[
        "numpy",
        "tqdm",
        "gensim",
        "keras",
        "tensorflow",
        "sklearn",
        "six",
        "gensim",
        "pandas",
        "h5py",
        "deepcut",
        # "keras_contrib==2.0.8"  # install manually
    ],
    # dependency_links=[
    #     'git+https://www.github.com/keras-team/keras-contrib.git#egg=keras-contrib-2.0.8'
    # ],
)
