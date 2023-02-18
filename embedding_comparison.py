import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel

class EmbeddingComparison:
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        self.words = ['apple', 'orange', 'banana', 'car', 'bus', 'train']
        self.embeddings = {
            'Word2Vec': None,
            'FastText': None,
            'Glove': None,
            'BOW': None,
            'TF-IDF': None,
            'BERT': None
        }
        self.word_vectors = None