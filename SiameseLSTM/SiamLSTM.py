import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow_core.python.keras import regularizers


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


def create_malstm_model(max_seq_length, embedding_dims, embeddings):
    # Parameters
    dropout_lstm = 0.23
    dropout_dense = 0.23
    regularizing = 0.002

    n_hidden = 300
    # Input layers
    left_input = layers.Input(shape=(max_seq_length,), dtype='int32')
    right_input = layers.Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = layers.Embedding(len(embeddings), embedding_dims,
                                       weights=[embeddings], input_length=max_seq_length, trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = layers.LSTM(n_hidden, dropout=dropout_lstm, kernel_regularizer=regularizers.l2(regularizing),
                              recurrent_dropout=dropout_lstm)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Concatenate the two question representations and the engineered features if they exist
    concatenated = layers.Concatenate()([left_output, right_output])
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)

    concatenated = layers.Dense(150, kernel_regularizer=regularizers.l2(regularizing), activation='relu')(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)

    concatenated = layers.Dense(70, kernel_regularizer=regularizers.l2(regularizing), activation='relu')(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)

    concatenated = layers.Dense(35, kernel_regularizer=regularizers.l2(regularizing), activation='relu')(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)

    output = layers.Dense(1, activation='sigmoid')(concatenated)

    return Model([left_input, right_input], output)
