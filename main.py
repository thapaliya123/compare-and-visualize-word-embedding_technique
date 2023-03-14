import os
import setup
from embedding_methods import word2vec, glove, bert
# get environment variables
pretrained_word2vec = os.getenv('pretrained_word2vec')
pretrained_glove = os.getenv('pretrained_glove')
pretrained_bert = os.getenv('pretrained_bert')
plot_save_dir = os.getenv('plot_save_dir')

WORD_CONTEXT_PAIR = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'),\
                      ('withdrawal', 'bank withdrawal')]


# Instantiate the Word EmbeddingComparison class and compare the embeddings
word2vec = word2vec.Word2Vec(pretrained_word2vec, plot_save_dir)
glove = glove.Glove(pretrained_glove, plot_save_dir)
bert = bert.BertEmbeddings(pretrained_bert, plot_save_dir)

def main():
    print('\n**Word2Vec EMBEDDINGS**')
    word2vec(WORD_CONTEXT_PAIR)

    print('\n**GLOVE EMBEDDINGS**')
    glove(WORD_CONTEXT_PAIR)

    print('\n**BERT EMBEDDINGS**')
    bert(WORD_CONTEXT_PAIR)

if __name__=='__main__':
    main()