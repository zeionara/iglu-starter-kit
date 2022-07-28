import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

class RandomClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length=128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        #TODO: Load weights


    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def clarification_required(self, instruction, gridworld_state):
        """
        Implements classifier for given instuction - whether a clarifying question is required or not
        Inputs:
            instruction - Single instruction string

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

        Outputs:
            0 or 1 - 0 if clarification is not required, 1 if clarification is required 

        """

        with torch.no_grad()
            encoded_dict = self.tokenizer.encode_plus(
                                instruction.lower(),
                                add_special_tokens = True, 
                                max_length = self.max_seq_length,           
                                pad_to_max_length = True,
                                truncation=True, 
                                return_attention_mask = True,   
                                return_tensors = 'pt',     
                        )
            
            inputs = torch.unsqueeze(torch.tensor(encoded_dict['input_ids']), dim=0)
            attention_mask = torch.unsqueeze(torch.tensor(encoded_dict['attention_mask']), dim=0)
            
            results = self.model(inputs, attention_mask=attention_mask)

        return results.logits.cpu().numpy()[0]