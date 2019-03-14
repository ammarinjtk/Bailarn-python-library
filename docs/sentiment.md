# Sentiment Analyzer
Sentiment Analyzer for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41) contains 5 main python files
- **constant.py**: Define all constant variables for other modules to use
- **metric.py**: Define evaluation metric for the model
- **model.py**: Define Keras model structure
- **callback.py**: Define Keras callback to fit the model
- **analyzer.py**: Define SentimentAnalyzer class containing main functions as below
    *   init (model_path=None, new_model=False, tag_index=None, embedding_matrix=None)
        - By default, if `tag_index` is None, then it will be created by `TAG_LIST` in `constant.py`.
        - `embedding_matrix` is used to initial weights for embedding layer in Keras model, if None the weights will be randomized.
        - If `new_model` is True, then the empty keras model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
    *	train (x_train, y_train, train_name='untitled', validation_split=0.1, epochs=1, batch_size=32)
        - Fit the model by the received parameters
        - `x_train`: contains char indices for every samples
        - `e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]`
        - `y_train`: contains label for every samples in term of k-hot encoding
        - `e.g. [ [[0,1], [1,0], …], … ]`
        - train_name is used for checkpoint creation
        - others are the parameters for fit function

    *	predict (x, decode_tag=True)
        - Receive word as input and transform it into input form of the model (X)
        - Then `decode_tag` will determine that the output will be probability lists for each of classes or the decoded one (label's name)
        ```
        >>> predict("ร้านนี้อาหารไม่อร่อย", decode_tag=False)
        [[0.01,0.99]]
        >>> predict("ร้านนี้อาหารไม่อร่อย", decode_tag=True)
        ['NEG']
        ```

    *	evaluate (x_true, y_true)
        - `x_true` and `y_true` are the same format as in train function
        - Evaluate the model by custom_metric from metric.py
        - By default, the precision, recall, f1_macro and f1_micro is calculated by sklearn
    *	save (model_path)
        - Save the trained model to the specific file path
