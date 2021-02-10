import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow_core.python.keras import regularizers


def create_malstm_features_model(max_seq_length, embedding_dims, embeddings, numb_engineered_features):
    # Parameters
    dropout_lstm = 0.23
    dropout_dense = 0.23
    regularizing = 0.002

    n_hidden = 300
    # Input layers
    left_input = layers.Input(shape=(max_seq_length,), dtype='int32')
    right_input = layers.Input(shape=(max_seq_length,), dtype='int32')
    engineered_features_input = layers.Input(shape=(numb_engineered_features,))

    # Embedding layer
    embedding_layer = layers.Embedding(len(embeddings), embedding_dims,
                                       weights=[embeddings], input_length=max_seq_length, trainable=False)
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = layers.LSTM(n_hidden, kernel_regularizer=regularizers.l2(regularizing), dropout=dropout_lstm,
                              recurrent_dropout=dropout_lstm, name="Siamese_LSTM")
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # One fully connected layer to transform the engineered features
    encoded_engineered = layers.Dense(70, activation='relu', name="FeatureDense")(engineered_features_input)

    # Concatenate the two question representations and the engineered features if they exist
    concatenated = layers.Concatenate()([left_output, right_output, encoded_engineered])
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)

    concatenated = layers.Dense(150, kernel_regularizer=regularizers.l2(regularizing), activation='relu',
                                name="ConcatenatedDense_1")(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization(name="BatchNorm1")(concatenated)

    concatenated = layers.Dense(70, kernel_regularizer=regularizers.l2(regularizing), activation='relu',
                                name="ConcatenatedDense_2")(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization(name="BatchNorm2")(concatenated)

    concatenated = layers.Dense(35, kernel_regularizer=regularizers.l2(regularizing), activation='relu',
                                name="ConcatenatedDense_3")(concatenated)
    concatenated = layers.Dropout(dropout_dense)(concatenated)
    concatenated = layers.BatchNormalization(name="BatchNorm3")(concatenated)

    output = layers.Dense(1, activation='sigmoid', name="Sigmoid")(concatenated)

    return Model([left_input, right_input, engineered_features_input], output)
