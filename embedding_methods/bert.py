import torch
from transformers import BertTokenizer, BertModel

class BertEmbeddings(VisualizeEmbeddings):
    '''
    Class to load Bert Tokenizer and Bert Model and generate Context aware Embeddings
    '''

    def __init__(self, bert_model_name: str, plot_save_dir: str) -> None:
        super().__init__(model_name='bert')
        self.bert_model_name = bert_model_name
        

    def load_model(self):
        '''
        Load pretrained BertModel and BertTokenizer

        Arguments:
            self: class object
        '''
        print('Loading pretrained BERT model and BERT tokenizers!!!')
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = BertModel.from_pretrained(self.bert_model_name)

    def forward_propagate(self, word, context):
        encoded_context = self.bert_tokenizer.encode_plus(context, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(encoded_context['input_ids'], attention_mask=encoded_context['attention_mask'])
            outputs.last_hidden_state[0][encoded_context['input_ids'].numpy().tolist()[0].index(self.bert_tokenizer.vocab[word])].numpy().reshape(768,)

        

    
