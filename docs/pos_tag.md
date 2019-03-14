# Part of Speech Tagger
Part of Speech Tagger for NLP Text Platform as a senior project for Computer Engineering, Chulalongkorn University (CP41) contains 5 main python files
- **constant.py**: Define all constant variables 
- **metric.py**: Define evaluation metric for the model
- **model.py**: Define Keras model structure, save and load model method.
- **callback.py**: Define Keras callback to fit the model
- **pos_tagger.py**: Define POStagger class containing main functions as below
    *   __init__( model_path=None, new_model=False, tag_index=None, embedding_matrix=None)
        - By default, if `tag_index` is None, then it will be use by `TAG_INDEXER` in `constant.py`.
        - `embedding_matrix` is used to initial weights for embedding layer in Keras model, if None the weights will be randomized.
        - If `new_model` is True, then the empty Keras model structure is defined
        - Otherwise, if `model_path` is defined then the model will be loaded from this path, while if `model_path` is undefined then our default trained model is loaded instead
    *	train ( x_true, y_true, train_name='untitled', validation_split=0.1,epochs=100, batch_size=32, learning_rate=0.001,shuffle=False, save_checkpoint_model=True)
        - Fit the model by the received parameters
        - `x_true`: contains word indices for every samples
        - `e.g. [ [word_index1, word_index2, …], [word_index1, word_index2], … ]`
        - `y_true`: contains label for every samples in shape (?,?,1)
        - `e.g. [ [[5], [1], …],[[7], [23], …] … ]`
        - `train_name` is used for checkpoint creation

        -`save_checkpoint_model` is th flag indies the model checkpoint will be saved when the epoch end on not
        - others are the parameters for fit function

    *	predict (x_vector, decode_tag = True)
        - Receive word index as input and transform it into tag 
        - Then `decode_tag` will determine that the output will be probability lists for each of classes or the decoded one (label's name)
   

    *	evaluate (x_true, y_true)
        - `x_true` and `y_true` are the same format as in train function
        - Evaluate the model by custom_metric from metric.py
        - By default, the precision, recall, f1_macro and f1_micro is calculated by sklearn
    *	save (filepath)
        - Save the trained model to the specific file path
        - The value error will be raised if the model is not trained or the file path is already exists
