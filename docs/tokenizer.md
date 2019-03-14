# Tokenizer
Tokenizer for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41) contains 5 main python files
- **constant.py**: Define all constant variables for other modules to use
- **metric.py**: Define evaluation metric for the model
- **model.py**: Define Keras model structure
- **callback.py**: Define Keras callback to fit the model
- **tokenizer.py**: Define Tokenizer class containing main functions as below
    *   init (char_index=None, tag_index=None, model_path=None, new_model=False)
        - `tag_index` is expected to be pos tags
        - By default, if `tag_index` is None, then it will be created by `TAG_LIST` in `constant.py`.
        - And if `char_index` is None, then it will be created by `CHAR_LIST` in `constant.py`
        - If `new_model` is True, then the empty keras model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
    *	train (x_train, y_train, train_name="untitled", validation_split=0.1, learning_rate=0.001, epochs=100, batch_size=32, shuffle=False)
        - Fit the model by the received parameters
        - `x`: contains char indices for every samples
        - `e.g. [ [char_index1, char_index2, …], [char_index1, char_index2], … ]`
        - `y`: contains label for every samples in term of one-hot encoding
        - by default, tag 1 is not expected to be the segmentation point and tag 0 is padding point, while others are POS tag (since our default Tokenizer model is joint-model between tokenizer and POS tagger tasks)
        - For example, in the sample_1, assume that segmentation point is at char_index2 and there are only 3 tags exists [ padding, unsegmented, pos_tag ], which means that tag for the first char is 1 and tag for the second char is 2 in form of one-hot
        - `e.g. [ [[0,1,0], [0,0,1], …], … ]`
        - train_name is used for checkpoint creation
        - others are the parameters for fit function

    *	predict (word, word_delimiter="|")
        - Receive word as input and transform it into input form of the model (X), then create and return token lists as output
        ```
        >>> predict("ฉันกินข้าว")
        [“ฉัน”, “กิน”, “ข้าว”]
        ```
    *	evaluate (x, y)
        - `x` and `y` are the same format as in train function
        - Evaluate the model by custom_metric from metric.py
        - By default, the precision, recall, f1_macro and f1_micro is calculated by sklearn
    *	save (filepath)
        - Save the trained model to the specific file path
        - The value error will be raised if the model is not trained or the file path is already exists
