import os
import setup
from embedding_methods import word2vec
# get environment variables
pretrained_word2vec = os.getenv('pretrained_word2vec')
plot_save_dir = os.getenv('plot_save_dir')

WORD_CONTEXT_PAIR = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'),\
                      ('withdrawal', 'bank withdrawal')]

# Instantiate the Word EmbeddingComparison class and compare the embeddings
word2vec = word2vec.Word2Vec(pretrained_word2vec, plot_save_dir)


def main():
    words, contexts = zip(*WORD_CONTEXT_PAIR)
    word2vec(WORD_CONTEXT_PAIR)

if __name__=='__main__':
    main()