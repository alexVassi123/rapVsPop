import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os, json
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint

def main():

    "Main function to train the rap-vs-pop model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True, help="Folder produced by datasplit")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="outputs/train",
                        help="Directory to write model and artefacts")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.json")) as fp:
        meta = json.load(fp)

    vocab_size    = meta["vocab_size"]
    max_length    = meta["max_length"]
    embedding_dim = meta["embedding_dim"]

    embedding_matrix = np.load(os.path.join(args.data_dir, "embedding_matrix.npy"))
    X_train_pad      = np.load(os.path.join(args.data_dir, "X_train_pad.npy"))
    y_train          = np.load(os.path.join(args.data_dir, "y_train.npy"))
        

    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.output_dir, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        X_train_pad, y_train,
        epochs=args.epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    model.save(os.path.join(args.output_dir, "model_final.h5"))


if __name__ == "__main__":
    main()