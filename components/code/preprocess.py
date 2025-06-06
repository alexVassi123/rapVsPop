# preprocess_data.py
import numpy as np
import pandas as pd
import argparse
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Preprocess text data for model training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with split CSVs")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Maximum vocabulary size for tokenizer")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension of word embeddings")
    parser.add_argument("--glove_file", type=str, default="data/glove.6B.100d.txt", help="Path to the GloVe embeddings file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed artifacts")
    args = parser.parse_args()

    # Load split data
    X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")["lyric"]
    X_test = pd.read_csv(f"{args.data_dir}/X_test.csv")["lyric"]
    y_train = pd.read_csv(f"{args.data_dir}/y_train.csv").values
    y_test = pd.read_csv(f"{args.data_dir}/y_test.csv").values

    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=args.max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=args.max_length, padding='post', truncating='post')

    # Create embedding matrix
    embedding_index = {}
    with open(args.glove_file, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = vector

    embedding_matrix = np.zeros((args.vocab_size, args.embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < args.vocab_size:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train_pad.npy", X_train_pad)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test_pad.npy", X_test_pad)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "embedding_matrix.npy", embedding_matrix)

    with open(out_dir / "tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "vocab_size": args.vocab_size,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
        }, f)

if __name__ == "__main__":
    main()
