import os
import setup
from embedding_methods import word2vec, glove
# get environment variables
pretrained_word2vec = os.getenv('pretrained_word2vec')
pretrained_glove = os.getenv('pretrained_glove')
plot_save_dir = os.getenv('plot_save_dir')

WORD_CONTEXT_PAIR = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'),\
                      ('withdrawal', 'bank withdrawal')]

# Instantiate the Word EmbeddingComparison class and compare the embeddings
word2vec = word2vec.Word2Vec(pretrained_word2vec, plot_save_dir)
glove = glove.Glove(pretrained_glove, plot_save_dir)

def main():
    words, contexts = zip(*WORD_CONTEXT_PAIR)
    print('\n**Word2Vec Embeddings**')
    word2vec(WORD_CONTEXT_PAIR)

    print('\n**GLOVE EMBEDDINGS**')
    glove(WORD_CONTEXT_PAIR)

if __name__=='__main__':
    main()