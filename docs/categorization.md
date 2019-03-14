# Text Categorization
Text Categorization for Thai Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41) contains 5 main python files
- **constant.py**: Define all constant variables for other modules to use
- **metric.py**: Define evaluation metric for the model
- **model.py**: Define Keras model structure
- **callback.py**: Define Keras callback to fit the model
- **categorization.py**: Define Categorization class containing main functions as below
    *   init (tag_index=None, embedding_matrix=None, model_path=None, new_model=False)
        - By default, if `tag_index` is None, then it will be created by `TAG_LIST` in `constant.py`.
        - `embedding_matrix` is used to initial weights for embedding layer in Keras model, if None the weights will be randomized.
        - If `new_model` is True, then the empty keras model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
    *	train (x_train, y_train, train_name="untitled", batch_size=100, learning_rate=0.001, validate_ratio=0.1,
            epochs=30, sensitive_learning=True)
        - Fit the model by the received parameters
        - `sensitive_learning` is used to apply Cost Sensitive Learning to adjust class_weight of the model.
        - `x`: contains char indices for every samples
        - `e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]`
        - `y`: contains label for every samples in term of k-hot encoding (supports Multi-label task)
        - `e.g. [ [[0,1,1], [1,0,1], …], … ]`
        - train_name is used for checkpoint creation
        - others are the parameters for fit function

    *	predict (x, threshold_selection=constant.DEFAULT_threshold_SELECTION_PATH, decode_tag=True)
        - Receive word as input and transform it into input form of the model (X)
        - Then `decode_tag` will determine that the output will be probability lists for each of classes or the decoded one (label's name)
        ```
        >>> predict("ช่วงนี้ AIS บริการไม่ดี", decode_tag=False)
        array([[0.36949787, 0.5046232 , 0.45457408]], dtype=float32)
        >>> predict("ช่วงนี้ AIS บริการไม่ดี", decode_tag=True)
        [['Mobile Operator', 'AIS', 'โทรศัพท์มือถือ']]
        ```
        - Note that `threshold_selection` is used to interpret the probability for each of classes, can be `Float` or `FilePath` is form of python dictionary {"class_k": Float}, for example
        ```
        $ vim ./threshold_selection.json
        {"class_0": 0.005, "class_1": 0.68, "class_2": 0.005}
        ```
    *	evaluate (x, y, threshold_selection=constant.DEFAULT_threshold_SELECTION_PATH)
        - `x` and `y` are the same format as in train function
        - `threshold_selection` is required to be the same as predict function
        - Evaluate the model by custom_metric from metric.py
        - By default, the precision, recall, f1_macro and f1_micro is calculated by sklearn
    *	save (filepath)
        - Save the trained model to the specific file path
        - The value error will be raised if the model is not trained or the file path is already exists
