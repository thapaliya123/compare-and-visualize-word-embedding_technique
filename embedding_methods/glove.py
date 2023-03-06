import numpy as np
import gensim
from embedding_methods.visualize_embeddings import VisualizeEmbeddings

class Glove(VisualizeEmbeddings):
    '''
    class to load pretrained glove model and perform some mathematical operations on words
    '''
    def __init__(self, pretrained_model_path: str, plot_save_dir: str) -> None:
        """
        Constructor

        Args:
            pretrained_model_path (str): Local path for pretrained glove text
        """
        super().__init__(model_name='glove', plot_save_dir = plot_save_dir)
        self.pretrained_model_path = pretrained_model_path

    def load_model(self):
        """      
        Load pretrained glove vector text from downloaded path location.

        Arguments:
            self: class object
        """

        try:
            print("Loading Pretrained Glove text from directory!!!")
            #load .txt file and loop through each line in it
            embeddings_dict = {}
            with open(self.pretrained_model_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    # update the dictionary with word and its correspoding word vectors
                    embeddings_dict[word] = vector
            return embeddings_dict
        except Exception as e:
            print(f"Error loading pretrained Glove text. Exception is:\n{e}")
            return None
    
    def get_embeddings(self, word):
        """
        get embeddings from the model passed
        """
        assert self.glove_model != None, "pretrained Glove text is not loaded!!!"

        word_vector = self.glove_model[word]
    
        return word_vector
        
    def __call__(self, word_context_pair):
        
        # load pretrained word2vec model
        self.glove_model = self.load_model()

        # unroll (word, context) pair
        word_list, context_list = zip(*word_context_pair)

        # get word2vec embeddings
        word_vector_arr = np.array([self.get_embeddings(word) for word in word_list])

        self.visualize_word_embeddings(word_vector_arr, word_context_pair)

        return word_vector_arr 

def main():
    glove_text_path = '/home/fm-pc-lt-125/Documents/personal/compare-and-visualize-word-embedding_technique/glove.6B.50d.txt'

    word_context_pair = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'), ('withdrawal', 'bank withdrawal')]
    words, contexts = zip(*word_context_pair)
    glove = Glove(glove_text_path, '')
    glove(word_context_pair)

if __name__ == '__main__':
    main()

    