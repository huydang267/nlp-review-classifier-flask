from pathlib import Path

from nltk.tokenize import RegexpTokenizer

_TOKENIZER = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
_STOPWORDS_PATH = Path(__file__).with_name("stopwords_en.txt")
with _STOPWORDS_PATH.open(encoding="utf-8") as f:
    _STOPWORDS = {w.strip() for w in f.read().splitlines() if w.strip()}


def preprocess(raw_text):
    if raw_text is None:
        return []
    tokens = _TOKENIZER.tokenize(str(raw_text))
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens
