import pandas as pd
import os
import numpy as np
import json
from sklearn.metrics import f1_score
from models.user_config import UserRanker, UserClassifer
from tqdm.auto import tqdm

def read_json_file(fname):
    with open(fname, 'r') as fp:
        d = json.load(fp)
    return d

def get_gridworld_state(index, instructions_df):
    state_file_path = instructions_df.loc[index].InitializedWorldPath
    state_file_path = os.path.join('./private_data', state_file_path)
    state = read_json_file(state_file_path)
    for drop_key in ['gameId', 'stepId', 'tape', 'clarification_question']:
        state.pop(drop_key)
    return state

def run_classification(classifier, instuctions_df, LocalEvalConfig):
    predictions = {}
    for index, instuction in tqdm(instuctions_df.InputInstruction.iteritems(), 
                                  total=len(instuctions_df),
                                  desc='Running classifier'):
        gridworld_state = get_gridworld_state(index, instuctions_df)
        res = classifier.clarification_required(instuction, gridworld_state)
        assert res in [0, 1], "Result of classfier should be 0 or 1"
        predictions[instuction] = int(res)
        
    with open(LocalEvalConfig.CLASSIFIER_RESULTS_FILE, 'w') as fp:
        json.dump(predictions, fp)



def run_ranking(ranker, instuctions_df, LocalEvalConfig):
    predictions = {}
    for index, instuction in tqdm(instuctions_df.InputInstruction.iteritems(),
                                  total=len(instuctions_df),
                                  desc='Running ranker'):
        gridworld_state = get_gridworld_state(index, instuctions_df)
        question_bank_path = os.path.join(LocalEvalConfig.DATA_FOLDER, 'question_bank.json')
        question_bank = read_json_file(question_bank_path)["question_bank"]

        res = ranker.rank_questions(instuction, gridworld_state, question_bank, 
                                    LocalEvalConfig.MAX_QUESTIONS_PER_INSTRUCTION)

        assert isinstance(res, list) or isinstance(res, tuple), "Output of ranker must be a list/tuple of strings"
        predictions[instuction] = list(res)
    
    with open(LocalEvalConfig.RANKER_RESULTS_FILE, 'w') as fp:
        json.dump(predictions, fp)


def evaluate(LocalEvalConfig):
    """
    Runs the local evaluation in same way as the aicrowd evaluator
    Both ranker and classifer will be run independently in this code
    Final evaluation code is the same as the evaluator
    """

    instuctions_df = pd.read_csv(os.path.join(LocalEvalConfig.DATA_FOLDER, 'ground_truth.csv'))
    
    # Run classfier
    classifier = UserClassifer()
    run_classification(classifier, instuctions_df, LocalEvalConfig)

    # Run ranker
    ranker = UserRanker()
    run_ranking(ranker, instuctions_df, LocalEvalConfig)

    # Load prediction files
    classifer_preds = read_json_file(LocalEvalConfig.CLASSIFIER_RESULTS_FILE)
    ranker_preds = read_json_file(LocalEvalConfig.RANKER_RESULTS_FILE)

    # Get classfication score
    classifier_gt = pd.Series(instuctions_df.IsInstructionClear.values, 
                              index=instuctions_df.InputInstruction).to_dict()
    cpreds, cgt = [], []
    for instruction, instruction_is_clear   in classifier_gt.items():
        cgt.append(int(instruction_is_clear.lower() == 'yes'))
        # if any instruction is not predicted, default value will be taken as 1
        cpreds.append(classifer_preds.get(instruction, 1)) 

    clariq_f1_score = f1_score(y_true=cgt, y_pred=cpreds)

    # Get ranker score
    unclear_rows = instuctions_df.dropna(subset=['ClarifyingQuestion'], inplace=False)
    ranker_gt = pd.Series(unclear_rows.ClarifyingQuestion.values, index=unclear_rows.InputInstruction).to_dict()
    inverse_rank_scores = []
    for instruction, clarifying_question  in ranker_gt.items():
        qpreds = ranker_preds[instruction]
        qpreds = qpreds[:LocalEvalConfig.MAX_QUESTIONS_PER_INSTRUCTION]
        if clarifying_question in qpreds:
            inverse_rank_scores.append(1/(qpreds.index(clarifying_question) + 1))
        else:
            inverse_rank_scores.append(0.0)
        
    clariq_mrr = np.mean(inverse_rank_scores)


    print("===================== Final scores =======================")
    print("F1 Score for Classifier", clariq_f1_score)
    print("MRR for Ranker", clariq_mrr)




if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        CLASSIFIER_RESULTS_FILE = './local-eval-classifier-results.json'
        RANKER_RESULTS_FILE = './local-eval-ranker-results.json'
        DATA_FOLDER = './private_data'
        MAX_QUESTIONS_PER_INSTRUCTION = 20
    
    evaluate(LocalEvalConfig)