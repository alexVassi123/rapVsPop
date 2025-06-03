import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import json, os
import argparse
import joblib

def main():
    "Main function to prepare data for training"
    # ---------- argument parsing ----------
    parser = argparse.ArgumentParser(description="Prepare data for rap‑vs‑pop training")
    parser.add_argument("--input_csv", type=str, default="data/train.csv", help="Path to the input CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--random_state", type=int, default=1234, help="Random seed for train‑test split")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Maximum vocabulary size for tokenizer")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length for padding")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension of word embeddings")
    parser.add_argument("--oov_token", type=str, default="<OOV>", help="Token for out‑of‑vocabulary words")
    parser.add_argument("--glove_file", type=str, default="data/glove.6B.100d.txt", help="Path to the GloVe embeddings file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the pre‑processed artefacts will be written")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    X = df["lyric"]
    y = df["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    vocab_size = args.vocab_size
    max_length = args.max_length
    embedding_dim = args.embedding_dim
    oov_token = args.oov_token

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    embedding_index = {}
    with open(args.glove_file, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = vector

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train_pad.npy", X_train_pad)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "embedding_matrix.npy", embedding_matrix)

    meta = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "embedding_dim": embedding_dim,

    }
    with open(out_dir / "meta.json", "w") as fp:
        json.dump(meta, fp)
                

if __name__ == "__main__":
    main()