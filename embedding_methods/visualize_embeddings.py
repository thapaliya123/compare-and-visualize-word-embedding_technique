import array
import pandas as pd
import utils

class VisualizeEmbeddings:
    def __init__(self, model_name: str) -> None:
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
        print(f'From super class: {self.model_name}')


    def visualize_word_embeddings(self, word_vectors, word_context_pair):
        word_vectors = utils.apply_pca(self.word_vectors)
        df_word_vectors = pd.DataFrame(word_vectors, colums=['x', 'y'])
        df_word_vectors['model'] = self.model_name
        df_word_vectors['word_to_context'] = [f"{word}:{context}" for word, context in word_context_pair]

        # Plot the embeddings in Scatter plot
        utils.create_plotly_scatter_plot(df_word_vectors, 'x', 'y', color='model', hover_name= 'word_context')