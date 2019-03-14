# Keyword expansion
Keyword expansion for Thai NLP Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41) contains 4 main python files
- **constant.py**: Define all constant variables 
- **metric.py**: Define evaluation metric for the model
- **keyword_expansion.py**: Define KeywordExpansion class containing main functions as below
    *   __init__( new_model=False, model_path=None, logging=True, tokenizer=deepcut.tokenize)
        - If `new_model` is True, then the empty Keras model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
        -`logging` indicate that log will be print or not
    *	train ( sentence, size=constant.VECTOR_DIM, sg=constant.CBOW_SG_FLAG,window=constant.WINDOW, min_count=constant.MIN_COUNT, workers=constant.WORKERS)
        - Fit the `gensim` model by the received parameters  
        - `sentence`: contains list of list of word
        - `e.g. [ [word1, word2, …, wordN], [word1, word2, …, wordM], … ]`
        - `size` is dimension of embed vector
        - `sg`: Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used
        - `window`: The maximum distance between the current and predicted word within a sentence. 
        - `min_count`: Ignores all words with total frequency lower than this.
        - `workers`: Use these many worker threads to train the mode

    *	predict (query_word, n_expand=constant.N_EXPAND)
        - Receive word as input and return list of word and similarity 
        - `query_word`: The word that want to expand
        - `n_expand`: number of word will be expanded 
   

    *	evaluate (query_word, expect_word, n_expand=constant.N_EXPAND)
        - evaluate F-1, precision, recall
        - `query_word`: The word that want to expand
        - `n_expand`: number of word will be expanded 
        - `expect_word`: list of expected word 
        
    *	save (filepath)
        - Save the trained model to the specific file path
        - The value error will be raised if the model is not trained or the file path is already exists
