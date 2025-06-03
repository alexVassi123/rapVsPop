import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    "Main function to prepare data for training"
    df = pd.read_csv("train.csv")  # Ensure this file is available during training in Azure ML

    X = df["lyric"]
    y = df["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    vocab_size = 5000
    max_length = 100
    embedding_dim = 100
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    embedding_index = {}
    with open("glove.6B.100d.txt", encoding="utf-8") as f:
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
                

if __name__ == "__main__":
    main()