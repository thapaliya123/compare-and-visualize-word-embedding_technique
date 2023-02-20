import numpy as np
from gensim.models import Word2Vec, FastText, KeyedVectors
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


    def train_classical_word_embedding(self, model_type, **kwargs):
        if model_type == 'Word2Vec':
            self.embeddings[model_type] = Word2Vec(self.corpus, **kwargs)
        
        elif model_type == 'FastText':
            self.embeddings[model_type] = FastText(self.corpus, **kwargs)
        
        elif model_type == 'Glove':
            self.embeddings[model_type] = KeyedVectors.load_word2vec_format(kwargs['file_path'], binary=False)

        else:
            raise ValueError(f'Unknown embedding type: {model_type}')
        
        return model_type
