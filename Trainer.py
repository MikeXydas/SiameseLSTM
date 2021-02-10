import shutil
import time
import pickle
import pandas as pd
import numpy as np

from tqdm.keras import TqdmCallback

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from sklearn import preprocessing

from SiameseLSTM.SiamLSTM import create_malstm_model
from SiameseLSTM.SiamLSTMwithFeatures import create_malstm_features_model
from SiameseLSTM.utils import create_dict_datasets, check_validation_acc, split_engineered_features_dataset
from SiameseLSTM.inference import create_output_file

if __name__ == "__main__":
    # Model variables
    batch_size = 1024
    n_epoch = 500
    use_engineered_features = False
    tensorboard_dir = 'storage\\logs\\'
    # model_checkpoint = "checkpoints/epoch_0470/cp.ckpt"
    model_checkpoint = ""
    delete_checkpoints_and_logs = True

    # CARE: To save time we have already transformed our texts from words to integers
    # and also created an embedding matrix (index -> embedding). In order to generate the
    # number representations you should use EmbeddingMatrix.ipynb and fix appropriately the
    # paths below
    submit_file = "storage/datasets/q2b/results/delete.csv"

    train_file = "storage/datasets/q2b/preprocessed/train_quora_clean.csv"
    test_file = "storage/datasets/q2b/preprocessed/test_quora_clean.csv"
    numb_representations_train_file = "storage/datasets/q2b/word_embeddings/numb_represantation_train.pkl"
    numb_representations_test_file = "storage/datasets/q2b/word_embeddings/numb_represantation_test.pkl"
    embedding_matrix_file = "storage/datasets/q2b/word_embeddings/embeddings_matrix.npy"
    engineered_features_train_file = "storage/datasets/q2b/features/train_features.csv"
    engineered_features_test_file = "storage/datasets/q2b/features/test_features.csv"

    # Deleting previous checkpoints and logs
    if delete_checkpoints_and_logs:
        try:
            shutil.rmtree('checkpoints/')
            print(">>> Deleted previous checkpoints")
            shutil.rmtree('storage/logs')
            print(">>> Deleted previous logs")
        except FileNotFoundError:
            print("No checkpoints or logs found")

    # Setting memory growth of GPU so as TF does not allocate all the available memory
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Read the texts
    print(">>> Reading the texts...", end='')
    clean_train_df = pd.read_csv(train_file)
    clean_test_df = pd.read_csv(test_file)
    print("Done")

    # Load the embeddings
    print(">>> Reading the embeddings...", end='')
    embeddings = np.load(embedding_matrix_file)
    with open(numb_representations_train_file, 'rb') as handle:
        numb_representation_train = pickle.load(handle)
    with open(numb_representations_test_file, 'rb') as handle:
        numb_representation_test = pickle.load(handle)
    print("Done")

    # Load the engineered features
    if use_engineered_features:
        print(">>> Reading the engineered features...", end='')
        engineered_features_train = np.array(pd.read_csv(engineered_features_train_file))
        X_feat_test = np.array(pd.read_csv(engineered_features_test_file))

        X_feat_train = engineered_features_train[:, :-1]
        y_feat_train = engineered_features_train[:, -1]

        normalizer = preprocessing.Normalizer().fit(X_feat_train)
        X_feat_train = normalizer.transform(X_feat_train)
        X_feat_test = normalizer.transform(X_feat_test)
        print("Done")
    else:
        X_feat_train, X_feat_test, y_feat_train = None, None, None

    embedding_dims = len(embeddings[0])

    print(">>> Creating the datasets...", end='')
    X_train, X_validation, X_test, Y_train, Y_validation, max_seq_length = \
        create_dict_datasets(clean_train_df, numb_representation_train, numb_representation_test)
    if X_feat_train is not None:
        X_features_train, X_features_val, Y_features_train, Y_features_validation, feat_size = \
            split_engineered_features_dataset(X_feat_train, y_feat_train)
    else:
        feat_size = 0
        X_features_train, X_features_val = None, None
    print("Done")

    print(">>> Starting training!")

    if use_engineered_features:
        malstm = create_malstm_features_model(max_seq_length, embedding_dims, embeddings, feat_size)
    else:
        malstm = create_malstm_model(max_seq_length, embedding_dims, embeddings)
    if model_checkpoint != "":
        malstm.load_weights(model_checkpoint)

    optimizer = Adam()

    malstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Tensorboard logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False,
                                                          histogram_freq=5)

    # Start training
    training_start_time = time.time()

    checkpoint_path = "checkpoints/epoch_{epoch:04d}/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    malstm.save_weights(checkpoint_path.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, period=5,
                                                     save_weights_only=True,
                                                     verbose=1)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=1e-2, patience=8000, verbose=1, restore_best_weights=True,
    )

    if feat_size == 0:
        malstm_trained = malstm.fit([X_train['left'], X_train['right'], ], Y_train, batch_size=batch_size, epochs=n_epoch,
                                    validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                                    callbacks=[cp_callback, early_stop_callback, TqdmCallback(verbose=1),
                                               tensorboard_callback],
                                    verbose=0)
    else:
        malstm_trained = malstm.fit(
            x=[X_train['left'], X_train['right'], X_features_train], y=Y_train,
            batch_size=batch_size, epochs=n_epoch,
            validation_data=([X_validation['left'], X_validation['right'], X_features_val],
                             Y_validation),
            callbacks=[cp_callback, early_stop_callback, TqdmCallback(verbose=1),
                       tensorboard_callback],
            verbose=0
        )

    print(">>> Training Finished!")

    # check_validation_acc(malstm, X_validation, Y_validation)
    print(">>> Predicting test results with the best validation model...", end='')
    if X_feat_test is None:
        create_output_file(malstm, [X_test['left'], X_test['right']], submit_file,
                           max_seq_length, embeddings, embedding_dims, from_path=False,
                           path_to_test=test_file)
    else:
        create_output_file(malstm, [X_test['left'], X_test['right'], X_feat_test], submit_file,
                           max_seq_length, embeddings, embedding_dims, from_path=False,
                           path_to_test=test_file)
    print("Done")
