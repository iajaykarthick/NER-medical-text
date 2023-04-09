import sys
sys.path.append('../')
from config import models, index_to_label, acronyms_to_entities, MAX_LENGTH

import tensorflow as tf

from scripts.utils import predict

from keras import backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision

def recall(y_true, y_pred):
    """Compute recall metric"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    """Compute f1-score metric"""
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    f1_score = 2 * ((_precision * _recall) / (_precision + _recall + K.epsilon()))
    return f1_score

def NER(model_name, text):
    # Print the arguments
    print("Model provided: ", models[model_name]['title'])
    model_path = models[model_name]['path']
    # Register the custom metric function
    tf.keras.utils.get_custom_objects()[precision.__name__] = precision
    tf.keras.utils.get_custom_objects()[recall.__name__] = recall
    tf.keras.utils.get_custom_objects()[f1_score.__name__] = f1_score
    model = tf.keras.models.load_model(model_path)
    
    predict(text, model, index_to_label, acronyms_to_entities, MAX_LENGTH)
