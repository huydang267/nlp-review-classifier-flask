# Data

The project uses a course-modified version of the public **Women's E-Commerce Clothing Reviews** dataset.

- Original source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- Original columns: Clothing ID, Age, Title, Review Text, Rating, Recommended IND, Positive Feedback Count, Division Name, Department Name, Class Name

## Course version (not included)

The course supplied a modified copy that is not redistributed here:

| File | Used by | Rows | Changes from the Kaggle original |
|---|---|---|---|
| `assignment3.csv` | `notebooks/` (Milestone I) | 19,662 | Subset of the original reviews, same 10 columns |
| `assignment3_II.csv` | `app/` (Milestone II) | 19,662 | Same reviews, plus two artificially generated columns: `Clothes Title` and `Clothes Description` |

To run the app, place `assignment3_II.csv` in this folder (`app/data/`). To run the notebooks, place `assignment3.csv` and `stopwords_en.txt` in `notebooks/`.

The Kaggle original does not contain the `Clothes Title` and `Clothes Description` columns, so the web app cannot run on it without adding those two columns.
