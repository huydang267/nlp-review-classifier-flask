import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for
from nltk.stem import PorterStemmer

from preprocess import preprocess

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "assignment3_II.csv"

with open(ROOT / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ROOT / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df = pd.read_csv(DATA_PATH)
products_df = df.drop_duplicates(subset="Clothing ID", keep="first")
departments = sorted(df["Department Name"].dropna().unique().tolist())

_stemmer = PorterStemmer()


def _stem_text(text: str) -> set[str]:
    return {
        _stemmer.stem(w.lower())
        for w in text.split()
        if len(w) > 1
    }


_search_corpus = [
    _stem_text(" ".join([
        str(row["Clothes Title"]),
        str(row["Clothes Description"]),
        str(row["Class Name"]),
        str(row["Department Name"]),
    ]))
    for row in products_df.to_dict("records")
]

app = Flask(__name__)


@app.context_processor
def inject_nav():
    return {"departments": departments}


@app.route("/")
def home():
    return render_template("home.html", products=products_df.to_dict("records"))


@app.route("/category/<department>")
def category(department):
    subset = products_df[products_df["Department Name"] == department]
    if subset.empty:
        abort(404)
    return render_template(
        "category.html",
        products=subset.to_dict("records"),
        department=department,
    )


@app.route("/item/<int:clothing_id>")
def item(clothing_id):
    match = products_df[products_df["Clothing ID"] == clothing_id]
    if match.empty:
        abort(404)
    item_info = match.iloc[0].to_dict()
    item_reviews = (
        df[df["Clothing ID"] == clothing_id]
        .fillna({"Title": "", "Review Text": ""})
        .assign(Rating=lambda x: x["Rating"].fillna(0).astype(int))
        .to_dict("records")
    )
    return render_template(
        "item.html",
        item=item_info,
        reviews=item_reviews,
        review_count=len(item_reviews),
    )


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("search.html", query="", products=[], count=None)

    query_stems = _stem_text(q)
    matched_rows = [
        products_df.iloc[i].to_dict()
        for i, prod_stems in enumerate(_search_corpus)
        if query_stems & prod_stems
    ]
    return render_template(
        "search.html",
        query=q,
        products=matched_rows,
        count=len(matched_rows),
    )


@app.route("/create-review", methods=["GET", "POST"])
def create_review():
    if request.method == "GET":
        clothing_id = request.args.get("clothing_id", type=int)
        if clothing_id is None:
            abort(404)
        match = products_df[products_df["Clothing ID"] == clothing_id]
        if match.empty:
            abort(404)
        return render_template(
            "create_review.html",
            product=match.iloc[0].to_dict(),
            prediction=None,
            error=None,
        )

    clothing_id = request.form.get("clothing_id", type=int)
    title = request.form.get("title", "").strip()
    review_text = request.form.get("review_text", "").strip()
    rating = request.form.get("rating", type=int)

    match = products_df[products_df["Clothing ID"] == clothing_id]
    if match.empty:
        abort(404)
    product = match.iloc[0].to_dict()

    error = None
    if not title:
        error = "Please enter a review title."
    elif not review_text:
        error = "Please enter your review."
    elif rating not in range(1, 6):
        error = "Please select a rating from 1 to 5."
    else:
        combined = preprocess(title) + preprocess(review_text)
        if not combined:
            error = "Your review text is too sparse. Please write at least one meaningful word."

    if error:
        return render_template(
            "create_review.html",
            product=product,
            prediction=None,
            error=error,
            form={"title": title, "review_text": review_text, "rating": rating},
        )

    joined = " ".join(combined)
    X = vectorizer.transform([joined])
    pred = int(model.predict(X)[0])
    proba = round(float(model.predict_proba(X)[0][pred]), 3)

    return render_template(
        "create_review.html",
        product=product,
        prediction=pred,
        proba=proba,
        error=None,
        form={"title": title, "review_text": review_text, "rating": rating},
    )


@app.route("/confirm-review", methods=["POST"])
def confirm_review():
    global df

    clothing_id = request.form.get("clothing_id", type=int)
    title = request.form.get("title", "").strip()
    review_text = request.form.get("review_text", "").strip()
    rating = request.form.get("rating", type=int, default=3)
    final_label = request.form.get("final_label", type=int)

    if clothing_id is None or final_label not in (0, 1):
        abort(400)

    match = products_df[products_df["Clothing ID"] == clothing_id]
    if match.empty:
        abort(404)
    product_row = match.iloc[0]

    new_row = pd.DataFrame([{
        "Clothing ID":             clothing_id,
        "Age":                     0,
        "Title":                   title,
        "Review Text":             review_text,
        "Rating":                  rating,
        "Recommended IND":         final_label,
        "Positive Feedback Count": 0,
        "Division Name":           product_row["Division Name"],
        "Department Name":         product_row["Department Name"],
        "Class Name":              product_row["Class Name"],
        "Clothes Title":           product_row["Clothes Title"],
        "Clothes Description":     product_row["Clothes Description"],
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return redirect(url_for("item", clothing_id=clothing_id))


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("500.html"), 500
