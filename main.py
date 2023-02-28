import os
import setup
from embedding_comparison import EmbeddingComparison

# get environment variables
pretrained_word2vec = os.getenv('pretrained_word2vec')

# Instantiate the Word EmbeddingComparison class and compare the embeddings
embedding_comparison = EmbeddingComparison(pretrained_word2vec=pretrained_word2vec)

def main():
    embedding_comparison.load_pretrained_word2vec()

if __name__=='__main__':
    main()