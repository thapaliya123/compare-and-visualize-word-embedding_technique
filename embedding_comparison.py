import shutil
import gzip
import gdown
from gensim.models import Word2Vec, FastText, KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel

from utils.helpers import delete_file

class EmbeddingComparison:
    def __init__(self, corpus=None, pretrained_word2vec=None, pretrained_glove=None) -> None:
        self.corpus = corpus
        self.pretrained_word2vec = pretrained_word2vec
        self.pretrained_glove = pretrained_glove
        self.words = ['apple', 'orange', 'banana', 'car', 'bus', 'train']
        self.embeddings = {
            'word2vec': None,
            'fasttext': None,
            'glove': None,
            'bow': None,
            'tf-idf': None,
            'bert': None
        }
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
        zip_file_name = 'GoogleNews-vectors-negative300.bin.gz'
        print("Downloading Pretrained Word2Vec from Google Drive!!!!!")
        url = f"https://drive.google.com/u/1/uc?id={gdrive_file_id}&export=download"
        gdown.download(url, zip_file_name)

        print("Unzip and save pretrained Word2Vec!!!")

        # Unzip the downloaded file
        with gzip.open(zip_file_name, 'rb') as f_in:
            with open('GoogleNews-vectors-negative300.bin', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print('Deleting word2vec zip file!!!!')
        # delete_file(zip_file_name)

    def load_pretrained_word2vec(self):
        """      
        
        Load pretrained word2vec model from downloaded path location.

        Arguments:
            self: class object
        """
        try:
            print("Loading Pretrained Word2Vec from directory!!!")
            file_name = self.pretrained_word2vec
            self.embeddings['word2vec'] = KeyedVectors.load_word2vec_format\
                                                    (file_name, binary=True)
        except FileNotFoundError:
            print("Miss pretrained Word2Vec file. please download 'GoogleNews-vectors-negative300.bin' \
                  and set environment variables")
            # self.download_pretrained_wordvec()
            # self.load_pretrained_word2vec()

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