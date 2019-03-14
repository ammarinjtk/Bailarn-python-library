# Word Embedder
Word embedder for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41)implemented by Gemsin(also see https://rare-technologies.com), this contains 2 main python files
- **constant.py**: Define all constant variables for other modules to use
- **word2vec.py**: Define Word2Vec class containing main functions as below
    *   init (model_path=None, new_model=False)
        - If `new_model` is True, then the empty gensim model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
    *	train (sentences)
        - Find all low frequency words happened less than k (defined in constant.py), then change it to be "UNK"
        - `sentences`: contains words for every samples(sentences)
        - `e.g. [ [word_1, word_2, ...], [word_1, word_2, ...], … ]`
        - The model can be trained repeatedly

    *	predict (word)
        - Receive word as input and return its vector representation in form of np.array, dimension size as defined in constant.py
        ```
        >>> predict("ฉัน")
        array([ 4.3708846e-02, -2.4344508e-01, -2.0937651e-01, ..., -8.1250496e-02,  3.1024747e-02,  4.8249800e-02], dtype=float32)
        ```
    *	evaluate (pairs)
        - This expects input to be list of pairs of word. `e.g. [("กิน", "รับประทาน"), ("นอน", "หลับ")]`
        - and will return the list of those pairs with similarity
    *	save (filepath)
        - Save the trained model to the specific file path
        - The value error will be raised if the model is not trained or the file path is already exists
