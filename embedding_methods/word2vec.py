import numpy as np
import gensim
from embedding_methods.visualize_embeddings import VisualizeEmbeddings

class Word2Vec(VisualizeEmbeddings):
    '''
    class to load word2vec model and perform some mathematical operations on words
    '''
    def __init__(self, pretrained_model_path: str, plot_save_dir: str) -> None:
        """
        Constructor

        Args:
            pretrained_model_path (str): Local path for pretrained word2vec model
        """
        super().__init__(model_name='word2vec', plot_save_dir = plot_save_dir)
        self.pretrained_model_path = pretrained_model_path
        self.model = None

    def load_model(self):
        """      
        Load pretrained word2vec model from downloaded path location.

        Arguments:
            self: class object
        """

        try:
            print("Loading Pretrained Word2Vec from directory!!!")
            file_name = self.pretrained_model_path
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
            return word2vec_model
        except Exception as e:
            print(f"Error loading pretrained Word2Vec models. Exception is:\n{e}")
            return None
    
    def get_embeddings(self, word):
        """
        get embeddings from the model passed
        """
        assert self.word2vec_model != None, "pretrained word2vec model is not loaded!!!"

        word_vector = self.word2vec_model[word]
    
        return word_vector
        
    def __call__(self, word_context_pair):
        
        # load pretrained word2vec model
        self.word2vec_model = self.load_model()

        # unroll (word, context) pair
        word_list, context_list = zip(*word_context_pair)

        # get word2vec embeddings
        word_vector_arr = np.array([self.get_embeddings(word) for word in word_list])

        self.visualize_word_embeddings(word_vector_arr, word_context_pair)

        return word_vector_arr
        

def main():
    word2vec_path = '../GoogleNews-vectors-negative300.bin'
    word2vec_path = '/home/fm-pc-lt-125/Documents/personal/compare-and-visualize-word-embedding_technique/GoogleNews-vectors-negative300.bin'
    word_context_pair = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'), ('withdrawal', 'bank withdrawal')]
    words, contexts = zip(*word_context_pair)

    # word2vec = Word2Vec(word2vec_path)
    # word2vec_model = word2vec.load_model()
    word2vec = Word2Vec(word2vec_path)
    print(word2vec(word_context_pair))
    
if __name__ == '__main__':
    main()

            
