import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "assignment3_II.csv"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"
MODEL_PATH = ROOT / "model.pkl"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    titles = df["Title"].fillna("").astype(str).tolist()
    reviews = df["Review Text"].fillna("").astype(str).tolist()
    y = df["Recommended IND"].astype(int).to_numpy()

    corpus = [" ".join(preprocess(t) + preprocess(r)) for t, r in zip(titles, reviews)]

    vectorizer = CountVectorizer(lowercase=False, token_pattern=r"[^\s]+")
    X = vectorizer.fit_transform(corpus)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X, y)

    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    print(
        f"rows: {len(corpus)}  vocab: {len(vectorizer.vocabulary_)}  "
        f"train acc: {model.score(X, y):.4f}"
    )


if __name__ == "__main__":
    main()
