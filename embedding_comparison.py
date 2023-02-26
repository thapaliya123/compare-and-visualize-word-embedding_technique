import os
import numpy as np
import gdown
import shutil
import gzip
from gensim.models import Word2Vec, FastText, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel


def delete_file(file_path):
    # delete existing files
    print('Deleting existing files!!!')
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
            print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

class EmbeddingComparison:
    def __init__(self, corpus=None, pretrained_word2vec=None, pretrained_glove=None) -> None:
        self.corpus = corpus
        self.words = ['apple', 'orange', 'banana', 'car', 'bus', 'train']
        self.embeddings = {
            'word2vec': None,
            'fasttext': None,
            'glove': None,
            'bow': None,
            'tf-idf': None,
            'bert': None
        }
        self.pretrained_word2vec = "models/GoogleNews-vectors-negative300.bin"
        self.pretrained_glove = pretrained_glove
        self.word_vectors = None

    def compute_bow_tfidf_vectors(self, vector_type):
        if vector_type == 'BOW':
            vectorizer = CountVectorizer()
        elif vector_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            raise ValueError("Invalid vector type. Supported vector types: BOW, TF-IDF")
        
        vectors = vectorizer.fit_transform(self.corpus).toarray()
        self.embeddings[vector_type] = vectors

        return vectors, vectorizer.vocabulary_
    
    def download_pretrained_wordvec(self):
        
        # Delete existing files before downloading new file
        delete_file(self.pretrained_word2vec)
    
        # Download the pretrained word2vec model
        gdrive_file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        
        print("Downloading Pretrained Word2Vec from Google Drive!!!!!")
        url = f"https://drive.google.com/u/1/uc?id={gdrive_file_id}&export=download"
        gdown.download(url, self.pretrained_word2vec)

        print("Unzip and save pretrained Word2Vec!!!")

        # Unzip the downloaded file
        with gzip.open(self.pretrained_word2vec, 'rb') as f_in:
            with open(self.pretrained_word2vec, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


    def load_pretrained_word2vec(self):
        """
        Load pretrained word2vec model from downloaded path location.
        """
        try:
            print("Loading Pretrained Word2Vec from directory!!!")
            self.embeddings['word2vec'] = KeyedVectors.load_word2vec_format(self.pretrained_word2vec, binary=True)
        except FileNotFoundError:
            print("Miss pretrained Word2Vec file.")
            self.download_pretrained_wordvec()
            self.load_pretrained_word2vec()

    def download_pretrained_glove(self):
        pass

    def load_pretrained_glove(self):
        pass

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

    def load_bert_model(self, model_name):
        self.embeddings['Bert'] = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def compute_bert_vectors(self):
        pass


def main():
    embedding_comp = EmbeddingComparison()
    embedding_comp.load_pretrained_word2vec()

    dog_vectors = embedding_comp.embeddings['word2vec']['dog']
    print(dog_vectors)
    
if __name__ == '__main__':
    main()