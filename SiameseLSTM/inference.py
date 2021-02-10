import pickle
import pandas as pd
import numpy as np

from SiameseLSTM.SiamLSTM import create_malstm_model
from SiameseLSTM.utils import create_dict_datasets


def create_output_file(model, X_test, outfile, max_seq_length, embeddings, embedding_dims, from_path=False,
                       path_to_test='../storage/datasets/q2b/test_without_labels.csv'):
    if from_path:
        loaded_model = create_malstm_model(max_seq_length, embedding_dims=embedding_dims, embeddings=embeddings)
        loaded_model.load_weights(model).expect_partial()
    else:
        loaded_model = model

    y_preds = loaded_model.predict(X_test)
    y_preds = np.round(y_preds)[:, 0].astype(int)

    test_ids_df = pd.read_csv(path_to_test, usecols=['Id'])

    results = {
        "Id": list(test_ids_df.Id),
        "Predicted": y_preds
    }

    results_df = pd.DataFrame.from_dict(results)

    results_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    # This main will work only in the case of not using the features
    # Read the texts
    print(">>> Reading the texts...", end='')
    clean_train_df = pd.read_csv('../storage/datasets/q2b/preprocessed/train_quora_clean.csv')
    clean_test_df = pd.read_csv('../storage/datasets/q2b/preprocessed/test_quora_clean.csv')
    print("Done")

    # Load the embeddings
    print(">>> Reading the embeddings...", end='')
    embeddings = np.load('../storage/datasets/q2b/word_embeddings/embeddings_matrix.npy', )
    with open('../storage/datasets/q2b/word_embeddings/numb_represantation_train.pkl', 'rb') as handle:
        numb_represantation_train = pickle.load(handle)
    with open('../storage/datasets/q2b/word_embeddings/numb_represantation_test.pkl', 'rb') as handle:
        numb_represantation_test = pickle.load(handle)
    print("Done")

    print(">>> Creating the datasets...", end='')
    X_train, X_validation, X_test, Y_train, Y_validation, max_seq_length = \
        create_dict_datasets(clean_train_df, clean_test_df, numb_represantation_train, numb_represantation_test)
    print("Done")

    embeddings_dim = len(embeddings[0])

    create_output_file(model='../checkpoints/epoch_0042/cp.ckpt',
                       X_test=[X_test['left'], X_test['right']],
                       outfile="../storage/datasets/q2b/results/test.csv",
                       max_seq_length=max_seq_length, embeddings=embeddings,
                       embedding_dims=embeddings_dim, from_path=True)

