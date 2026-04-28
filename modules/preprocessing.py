import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    """
    Text preprocessing module for text classification.

    Steps:
    1. Lowercase
    2. Remove non-letter characters
    3. Normalize whitespace
    4. Tokenize by whitespace
    5. Remove stopwords
    6. Optional stemming
    """

    def __init__(self, use_stemming=False, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        # Download stopwords if missing
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        text = str(text)

        # 1. Lowercase
        text = text.lower()

        # 2. Remove numbers, punctuation, special characters
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # 3. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 4. Tokenization by whitespace
        tokens = text.split()

        # 5. Remove short tokens
        tokens = [w for w in tokens if len(w) >= 2]

        # 6. Remove stopwords
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]

        # 7. Optional stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(w) for w in tokens]

        return " ".join(tokens)

    def transform(self, texts):
        return [self.clean_text(text) for text in texts]
