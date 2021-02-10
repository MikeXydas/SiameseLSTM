import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_dict_datasets(clean_train_df, numb_represantation_train, numb_represantation_test, seed=1212, val_ratio=0.2):
    max_seq_length = 30

    # Split to train validation
    validation_size = int(val_ratio * len(clean_train_df))
    training_size = len(clean_train_df) - validation_size

    X_train_Q1 = [t[0] for t in numb_represantation_train]
    X_train_Q2 = [t[1] for t in numb_represantation_train]
    X_test_Q1 = [t[0] for t in numb_represantation_test]
    X_test_Q2 = [t[1] for t in numb_represantation_test]

    results = {
        "Q1": X_train_Q1,
        "Q2": X_train_Q2
    }
    X = pd.DataFrame.from_dict(results)
    Y = clean_train_df[['IsDuplicate']]

    results = {
        "Q1": X_test_Q1,
        "Q2": X_test_Q2
    }
    X_test = pd.DataFrame.from_dict(results)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                    random_state=seed)

    # Split to dicts
    X_train = {'left': X_train.Q1, 'right': X_train.Q2}
    X_validation = {'left': X_validation.Q1, 'right': X_validation.Q2}
    X_test = {'left': X_test.Q1, 'right': X_test.Q2}

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    return X_train, X_validation, X_test, Y_train, Y_validation, max_seq_length


def split_engineered_features_dataset(X_feat, y_feat, seed=1212, val_ratio=0.2):
    validation_size = int(val_ratio * len(X_feat))
    X_features_train, X_features_val, Y_features_train, Y_features_validation = \
        train_test_split(X_feat, y_feat, test_size=validation_size, random_state=seed)

    return X_features_train, X_features_val, Y_features_train, Y_features_validation, X_feat.shape[1]


def check_validation_acc(model, X_validation, y_validation):
    y_preds = np.round(model.predict([X_validation['left'], X_validation['right']]))[:, 0].astype(int)

    print(accuracy_score(y_validation, y_preds))