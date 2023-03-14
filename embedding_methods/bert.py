import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from embedding_methods.visualize_embeddings import VisualizeEmbeddings

class BertEmbeddings(VisualizeEmbeddings):
    '''
    Class to load Bert Tokenizer and Bert Model and generate Context aware Embeddings
    '''

    def __init__(self, bert_model_name: str, plot_save_dir: str) -> None:
        super().__init__(model_name='bert', plot_save_dir=plot_save_dir)
        self.bert_model_name = bert_model_name
        
    def load_model(self):
        '''
        Load pretrained BertModel and BertTokenizer

        Arguments:
            self: class object
        '''
        print('Loading pretrained BERT model and BERT tokenizers!!!')
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)

    def forward_propagate(self, word, context):
        encoded_context = self.bert_tokenizer.encode_plus(context, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(encoded_context['input_ids'], attention_mask=encoded_context['attention_mask'])
            outputs = outputs.last_hidden_state[0][encoded_context['input_ids'].numpy().tolist()[0].index(self.bert_tokenizer.vocab[word])].numpy().reshape(768,)
            
        return outputs

    def __call__(self, word_context_pair):
        # load pretrained bert model
        # including bert_tokenizer, bert_model
        self.load_model()

        # forward propagate
        # get embeddings
        word_vector_arr = np.array([self.forward_propagate(word, context) for word, context in word_context_pair])

        # apply pca
        # visualize word vectors
        self.visualize_word_embeddings(word_vector_arr, word_context_pair)

        return word_vector_arr
    

def main():
    word_context_pair = [('bank', 'river bank'), ('bank', 'bank account'), ('deposit', 'bank deposit'), ('withdrawal', 'bank withdrawal')]
    plot_save_dir = '/home/fm-pc-lt-125/Documents/personal/compare-and-visualize-word-embedding_technique/static/plots/'
    bert = BertEmbeddings('bert-base-uncased', plot_save_dir)
    print('Generating Bert Embeddings Vector!!!')
    bert(word_context_pair)
if __name__ == '__main__':
    main()

        

    
