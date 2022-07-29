import numpy as np
import torch
from transformer_rankers.utils import utils
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.data.data_collator import default_data_collator
from torch.utils.data import Dataset, DataLoader
from transformer_rankers.trainers import transformer_trainer
from transformers.data.processors.utils import InputFeatures

class SimpleDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index]


class BERTRanker:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        #TODO: Load weights
        self.model.to(self.device)

        self.trainer = transformer_trainer.TransformerTrainer(model=self.model,
                            train_loader=None,
                            val_loader=None, test_loader=None,
                            num_ns_eval=1, task_type="classification", 
                            tokenizer=self.tokenizer,
                            validate_every_epochs=1, num_validation_batches=-1,
                            num_epochs=1, lr=5e-7, sacred_ex=None)

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def rank_questions(self, instruction, gridworld_state, question_bank):
        """
        Implements the ranking function for a given instruction
        Inputs:
            instruction - Single instruction string, may or may not need any clarifying question
                          The evaluator may pass questions that don't need clarification, 
                          But only questions requiring clarifying questions will be scored

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

            question_bank - List of clarifying questions to rank

        Outputs:
            ranks - A sorted list of questions from the question bank
                    Such that the first index corresponds to the best ranked question

        """

        examples = [(instruction, question) for question in question_bank]
        batch_encoding = self.tokenizer.batch_encode_plus(examples, 
                max_length=self.max_seq_len, pad_to_max_length=True)

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=0)
            features.append(feature)
    
        dataset = SimpleDataset(features)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, 
                                collate_fn=default_data_collator)



        _, _, softmax_output = self.trainer.predict(dataloader)
        softmax_output_by_query = utils.acumulate_list(softmax_output[0], 1)

        ranks = np.argsort(softmax_output_by_query[0])
        ranked_question_list = np.array(question_bank)[ranks]
        return list(ranked_question_list)