import array
import utils

class VisualizeEmbeddings:
    def __init__(self, model_name: str, word_vectors: array, word_contexts: list) -> None:
        """
        Initializes class object with desired variables

        Args:
            model_name (str): name of embeddings model
                    >>> e.g. Word2Vec, Gensim, Bert, etc
            word_vectors (array): 2D array denoting embeddings in n dimensions
            word_contexts (list): list of tuple denoting (word, context) pairs
                    >>> e.g [('bank', 'river bank'), ('bank', 'bank withdrawal')]
        """
        self.model_name = model_name
        self.word_vectors = word_vectors
        self.word_contexts = word_contexts

    def visualize_word_embeddings(self):
        word_vectors = utils.apply_pca(self.word_vectors)

        # Plot the embeddings in Scatter plot
        utils.create_plotly_scatter_plot()