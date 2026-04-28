import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Optional dependency: gensim is only needed for Word2Vec
try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None

import torch
from transformers import BertTokenizer, BertModel


def extract_traditional(train_texts, test_texts, method="tfidf", max_features=5000):
    if method == "bow":
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    return X_train, X_test


def extract_word2vec(train_tokens, test_tokens, vector_size=100):
    if Word2Vec is None:
        raise ImportError(
            "gensim is not installed. Please install it with: !pip install gensim"
        )

    model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4
    )

    def get_avg_vector(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

    X_train = np.array([get_avg_vector(t) for t in train_tokens])
    X_test = np.array([get_avg_vector(t) for t in test_tokens])

    return X_train, X_test


def extract_bert(texts, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []

    texts = list(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)
