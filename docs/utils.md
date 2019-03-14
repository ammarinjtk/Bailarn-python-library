# Utils
Utils for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University so that you can use every NLP task smoothly.

- **utils.py**: Define all useful util functions as below
    - Text class: object to contain the Text sample information
        - Args
            - path: contians file path, e.g.`./data/BEST_mock/T00129.xml.stripped.txt`
            - filename: contains only filename without full path, e.g.`T00129.xml.stripped.txt`
            - content: contains the content from that file, e.g.`เฒ่า/NN/O|วัย/NN/O|72/CD/MEA_B|`

    * TextCollection class: object to contain the Corpus by reading .txt filepath
        * Args
            - corpus_directory
            -	tokenize_function
            -	word_delimiter: `|` by default as BEST corpus pattern
            -	tag_delimiter: `/` by default as BEST corpus pattern
            -	tag_dictionary: `{'word': 0, 'pos': 1, 'ner': 2}` by default as BEST corpus pattern
            -	__corpus: Contains all text classes from corpus directory
        * Functions
            -	_load
                - Preprocess string, if tokenize_function is defined then this string will be tokenized
                - Create text class, then add it into `__corpus`
            -	_preprocessing (content): Handle dirty text content such as removing new line, trimming white space and replacing name entity symbol with empty.
            -	add_text (content): Add raw text into text collection manually
            -	get_label_from_file (file_path): Read answers from a `.lab` file (label y) corresponding to the `.txt` file (input x)
                - To be used for sentiment and categorization tasks
            -	count (): count all text classes in the text collection
            -	get_file_name (index): get file name from the selected text class by index
            -	get_path (index): get path from the selected text class by index
            -	get_content (index): get file content from the selected text class by index
            -	get_text (index): get selected `Text` class from index
            -	get_token_list (index): get all tokens from selected text class
                - If tokenize_function is defined, return all tokens from this function in case of sentiment and categorization tasks.
                - If word and tag delimiter are defined, words will be separated by word and tag delimiter, which are BEST_2010 pattern, in case of ner, posm tokenization and word_embedding tasks.
                - Otherwise, words will be separated by only word delimiter, which are BEST_data (BEST I Corpus) in case of tokenization and word_embedding tasks
    * InputCollection class: object to contain input for every NLP task
        * Args
            * x: `e.g. array([ 69, 233, 363, 133, 358])`
            * y: `e.g. array([[1], [1], [20], [11], [13]])`
            * readable_x: `e.g. ['เฒ่า', 'วัย', '72', 'ร้อง', 'ถูก']`
            * readable_y: `e.g. ['NN', 'NN', 'CD', 'VV', 'AUX']`
    * build_input (corpus, word_index, tag_index, sequence_length, target, for_train): generate vector collection to be ready-to-use input for the model
        * Args
            * corpus: TextCollection object
            * word_index: `e.g. {"ฉัน": 0, "กิน": 1, "UNK": 2}`
            * tag_index
            * sequence_length: input sequence length for padding.
            * target: string value for specific NLP task, e.g. `[tokenizer, pos, ner, categorization, sentiment]`
            * for_train: boolean, if True then x and y are expected to return, otherwise only x is expected.
    * pad (x, sequence_length, pad_with, mode)
        * pad the input for the model, there are 2 modes for this padding function
        * `mode = 0`: padding data by cutting exceed data into a new data sample
        * `mode = 1`: padding data by removing exceed data
        ```
        >>> pad(x=[1, 1, 1, 1], sequence_length=2, pad_with=0, mode=0)
        [[1, 1], [1, 1]]
        >>> pad(x=[1, 1, 1, 1], sequence_length=2, pad_with=0, mode=1)
        [[1, 1]]
        ```
    * generate_x_y(**kwargs): will be called by build_input
        * in case of sentiment and categorization tasks:
        * `x`: contains word indices for every samples (txt files)
        * `e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]`
        * `y`: contains label (lab files) in term of k-hot encoding
        * For example, sample_1 is class A, sample_2 is class A and B and all possible classes are A, B and C 
        * `e.g. [ [True, False, False], [True, True, False], ... ]`
        * in case of tokenization task:
        *	x: contains char indices for every samples
        *	`e.g. [ [char_index1, char_index2, …], [char_index1, char_index2], … ]`
        *	y: contains label for every samples in term of one-hot encoding
        *	by default, tag 1 is not expected to be the segmentation point and tag 0 is padding point, while others are POS tag (since our default Tokenizer model is joint-model between tokenizer and POS tagger tasks)
        *	For example, in the sample_1, assume that segmentation point is at char_index2 and there are only 3 tags exists [ padding, unsegmented, pos_tag ], which means that tag for the first char is 1 and tag for the second char is 2 in form of one-hot
        *	`e.g. [ [[0,1,0], [0,0,1], …], … ]`
        *	in case of ner task:
        *	x: contains word indices for every samples
        *	e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]
        *	y: contains label for every samples in term of one-hot encoding
        *	For example, in sample_1, the first word is tag 1, the second word is tag 2
        *	e.g. [ [[1,0], [0,1], …], … ]
        *	in case of pos task
        *	x: contains word indices for every samples
        *	e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]
        *	y: contains label for every samples
        *	e.g. [ [1, 2, …], [2, 3, …] ]
        *	For all cases, if word or char is unseen in word_index dictionary then the index for UNK is assigned instead.

    * generate_x(**kwargs): will be called by build_input, same as generate_x_y

    * build_tag_index (lst, start_index=1, reverse=False): create tag indexer by receiving list as input, and return its indexer as dict

    * build_word_index (text_collection, word2vec_vocab=None): create word indexer for all words
        * text_collection: TextCollection object to get all possible words in corpus
        * word2vec_vocab: python dictionary of all words from `word_embedder` vocabulary, if `word2vec_vocab` is None then the words from vocab will be excluded.

    * get_embedding_matrix (word2vec_model, word_index, fasttext=False): create embedding_matrix to initialize the embedding layer weights.
        * word2vec_model: expect word_embeddeing model implemented by [**Gensim**](https://rare-technologies.com)
        * word_index: python dictionary created by `build_word_index`
        * fasttext: support 2 word_embedding models which are Word2Vec and FastText.
        * Example of output can be `array([[ 4.37088460e-02, -2.43445083e-01, -2.09376514e-01, ...,
        -8.12504962e-02,  3.10247466e-02,  4.82497998e-02]])`, same size as word_index

    * clean_sentence (cleaned_sentence)
        * remove space(" ") and newline("\n")
        * clean all number to be zero(0)
        * remove entity symbol such as `<NE>`, `</AB>` and `<POEM>`

    * find_low_freq_words (all_sentences, threshold=5): used in `word_embedder` model to find all low frequency word to handle `UNK`

    * save_labels (prediction, path)
    * save_word_tag (x, prediction, target,path)




