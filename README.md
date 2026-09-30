# NLP Review Classifier with Flask

A text classification pipeline that predicts whether a clothing review recommends the product, deployed in a Flask shopping website that labels new reviews automatically.

## Business problem

An online clothing retailer collects thousands of free-text reviews. Each review carries a "recommended" flag that drives product recommendations for other shoppers, but customers do not always set it consistently. The retailer needs (1) a model that infers the recommendation label from the review text alone, and (2) a website where shoppers can browse products, read reviews and submit new reviews that are labelled automatically, with the option to correct the label before it is saved.

## Data

| Item | Detail |
|---|---|
| Source | [Women's E-Commerce Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |
| Version used | Course-modified version: 19,662 reviews, 1,095 products. The Milestone II file adds two artificially generated columns (`Clothes Title`, `Clothes Description`). |
| Modelling set | 19,652 reviews (10 reviews are empty after preprocessing and are dropped) |
| Target | `Recommended IND`: 81.82% recommended, 18.18% not recommended |
| Access | The course files are not redistributed. See [app/data/README.md](app/data/README.md). |

## Method

1. **Text preprocessing (Task 1).** Tokenised `Review Text` with the regex `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`, lowercased, removed single-character tokens and 570 stopwords, removed 6,734 words that occur only once in the corpus, and removed the 20 words with the highest document frequency. The resulting vocabulary has 7,529 tokens.
2. **Feature representations (Task 2).** Built three document representations: Count bag-of-words (19,652 × 7,529, sparse), mean-pooled GloVe 100d embeddings, and TF-IDF weighted GloVe 100d embeddings.
3. **Evaluation design (Task 3).** Trained Logistic Regression (`class_weight="balanced"`, `lbfgs`, `max_iter=1000`, `random_state=42`) under 5-fold stratified cross-validation. Macro F1 is the primary metric and is compared with a naive predict-all-recommended baseline.
4. **Q1: language model comparison.** Compared the three representations on review text only.
5. **Q2: does more information help?** Compared title only, review only, and title plus review. Title plus review was built in two ways: Approach A (concatenated tokens with one shared vocabulary) and Approach B (separate title and review vectorisers, stacked side by side).
6. **Model training for deployment.** `train_model.py` fits a Count vectoriser and a balanced Logistic Regression on all 19,662 reviews using one shared vocabulary (Approach A). The notebook preferred Approach A on parsimony because its score is within one standard deviation of Approach B.
7. **Web application (Milestone II).** The Flask app serves 1,095 products across 6 departments, with item pages that list all reviews and a stemmed keyword search. A new review is preprocessed with the same pipeline as training, classified, and shown with the predicted label and confidence. The shopper can accept or override the label, and the review is then appended to the dataset and appears on the item page.

Portfolio version: train_model.py now trains on the review Title (as the app classifies it) rather than the generated Clothes Title used in the original submission.

## Results

5-fold stratified cross-validation, Logistic Regression (from `notebooks/task2_3.ipynb`):

| Configuration | Macro F1 (mean ± std) | Accuracy |
|---|---|---|
| Naive baseline (predict all recommended) | 0.4500 | 0.8181 |
| Review only, Weighted GloVe | 0.6521 ± 0.0124 | 0.7262 |
| Review only, Unweighted GloVe | 0.6713 ± 0.0112 | 0.7455 |
| Title only, Count BOW | 0.7558 ± 0.0048 | 0.8261 |
| Review only, Count BOW | 0.7744 ± 0.0065 | 0.8511 |
| Title + Review, Count BOW (Approach A) | 0.8144 ± 0.0034 | 0.8797 |
| **Title + Review, Count BOW (Approach B)** | **0.8215 ± 0.0075** | **0.8855** |

Key findings:

- Count BOW outperformed both GloVe representations on review text (0.7744 vs 0.6713 and 0.6521). Averaging embeddings cancels opposing sentiment words, while BOW learns a separate weight for each token.
- Adding the review title improved macro F1 from 0.7744 to 0.8215. Titles are short but carry concentrated sentiment that the review pipeline filters out.
- Best model (Approach B) out-of-fold confusion matrix: 2,820 true negatives, 755 false positives, 1,496 false negatives and 14,581 true positives.

## Repo structure

```
nlp-review-classifier-flask/
├── notebooks/
│   ├── task1.ipynb            # Text preprocessing and vocabulary (outputs kept)
│   └── task2_3.ipynb          # Feature representations and model comparison (outputs kept)
├── app/
│   ├── app.py                 # Flask routes: browse, category, item, search, create and confirm review
│   ├── preprocess.py          # Shared preprocessing used by training and inference
│   ├── train_model.py         # Fits the vectoriser and model, writes model.pkl and vectorizer.pkl
│   ├── stopwords_en.txt       # 570-word stopword list used by preprocess.py
│   ├── requirements.txt
│   ├── data/
│   │   └── README.md          # Dataset source and placement instructions
│   ├── templates/             # Jinja2 templates (base, home, category, item, search, create_review, about, 404, 500)
│   └── static/                # CSS and site images
└── README.md
```

## How to run

Use Python 3.10 to 3.13. On Python 3.14 the pinned numpy, pandas and scikit-learn versions have no pre-built wheels, so pip compiles them from source.

**Web app**

```bash
git clone https://github.com/huydang267/nlp-review-classifier-flask.git
cd nlp-review-classifier-flask/app

# Place the course file assignment3_II.csv in app/data/ (see app/data/README.md)

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py          # creates model.pkl and vectorizer.pkl
flask run                      # open http://127.0.0.1:5000
```

**Notebooks**

```bash
cd notebooks
# Place assignment3.csv and stopwords_en.txt (copy from app/) in notebooks/
pip install pandas numpy nltk scikit-learn scipy matplotlib seaborn gensim jupyter
jupyter notebook task1.ipynb   # run task1 first: it writes processed.csv and vocab.txt
```

`task2_3.ipynb` downloads the GloVe 100d vectors (about 130 MB) through `gensim` on its first run.

A screen-recorded demo of the website can be provided on request.

## Context

Course project, RMIT University (Melbourne campus, cross-campus semester), COSC2815 Advanced Programming for Data Science, Semester 1 2026.
