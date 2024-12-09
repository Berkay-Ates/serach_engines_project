import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models import FastText


class EmbeddingMethods:
    def __init__(self) -> None:
        pass

    def albert_embeddings(self, data):
        print("albert_running")
        model_name = "albert-base-v2"
        embedder = SentenceTransformer(model_name)
        return embedder.encode(data)

    def bert_embedding(self, data):
        print("bert_running")
        # Model ve tokenizer yükleme
        model_name = "bert-base-multilingual-cased"
        embedder = SentenceTransformer(model_name)
        return embedder.encode(data)

    def distil_bert_embedding(self, data: list):
        print("distil-bert-running")
        embedder = SentenceTransformer("distilbert-base-nli-mean-tokens")
        return embedder.encode(data)

    def tf_idf_embedding(self, data: list):
        print("tf-idf-running")
        embedder = TfidfVectorizer()
        return embedder.fit_transform(data).toarray()

    def bag_of_words_embedding(self, data: list):
        print("bag_of_word_running")
        embedder = CountVectorizer()
        return embedder.fit_transform(data).toarray()

    def word2vec_embedding(self, data: list):
        print("word2Vec Running")
        embed_results = []
        text = pd.DataFrame(data, columns=["text"])
        text = text["text"].apply(lambda x: simple_preprocess(x))

        model = Word2Vec(vector_size=128, window=5, min_count=1, workers=4)
        model.build_vocab(text, progress_per=1000)
        model.train(text, total_examples=model.corpus_count, epochs=model.epochs)

        data_list = text.values.tolist()
        for sentence in data_list:
            res = np.zeros(128)  # Numpy array olarak sıfırlanmış bir vektör
            for word in sentence:
                res += model.wv[word]
            res /= len(sentence)
            embed_results.append(res.tolist())  # Tekrar listeye çevirerek ekleme
        return embed_results

    def __run_function__(self, func_name, param):
        func = getattr(self, func_name)
        return func(param)
