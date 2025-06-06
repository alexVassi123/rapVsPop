import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Split data into training and test sets")
    parser.add_argument("--input_csv", type=str, default="data/train.csv", help="Path to the input CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--random_state", type=int, default=1234, help="Random seed for train-test split")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split data")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    X = df["lyric"]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    os.makedirs(args.output_dir, exist_ok=True)
    X_train.to_csv(f"{args.output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{args.output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{args.output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{args.output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    print("Starting datasplit step...")
    main()
