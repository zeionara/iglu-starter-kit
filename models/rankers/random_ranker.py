import numpy as np

class RandomRanker:
    def __init__(self):
        pass

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def rank_questions(self, instruction, gridworld_state, question_bank, max_questions_per_instruction):
        """
        Implements the ranking function for a given instruction
        Inputs:
            instruction - Single instruction string, may or may not need any clarifying question
                          The evaluator may pass questions that don't need clarification, 
                          But only questions requiring clarifying questions will be scored

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

            question_bank - List of clarifying questions to rank
                            

            max_questions_per_instruction - The maximum number of clarifying questions you need to return, 
                                            any more than this number will not be scored
                                            This is expected to be 20 (but may change during the competition)

        Outputs:
            ranks - A sorted list of questions from the question bank
                    Such that the first index corresponds to the best ranked question

        """

        np.random.shuffle(question_bank)
        ranked_question_list = question_bank[:max_questions_per_instruction]
        return ranked_question_list