import gensim

class Word2Vec:
    '''
    class to load word2vec model and perform some mathematical operations on words
    '''
    def __init__(self, pretrained_model_path: str) -> None:
        """
        Constructor

        Args:
            pretrained_model_path (str): Local path for pretrained word2vec model
        """
        self.pretrained_model_path = pretrained_model_path

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

def main():
    word2vec_path = '../GoogleNews-vectors-negative300.bin'
    sample_word = 'bank'
    word2vec = Word2Vec(word2vec_path)
    word2vec_model = word2vec.load_model()
    print("generating embeddings from word2vec!!!")
    print(word2vec_model[sample_word])

if __name__ == '__main__':
    main()

            
