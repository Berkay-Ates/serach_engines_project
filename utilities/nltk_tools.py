from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import pandas as pd

stemmer = SnowballStemmer(language="english")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_stopwords(text):
    text = str(text)
    words = word_tokenize(text.lower())
    return " ".join(word for word in words if word not in stop_words)


def snowball_stemming(text):
    text = str(text)
    return " ".join(stemmer.stem(word) for word in text.split())


def wordnet_lemmatizer(text):
    text = str(text)
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())
