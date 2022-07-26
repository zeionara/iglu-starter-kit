## How to write your own models?

We recommend that you place the code for all your agents in the `models` directory (though it is not mandatory). All your submissions should contain a Classifer and a Ranker. We have added random classifier and ranker examples in `classifiers/random_classifier.py` and `rankers/random_ranker.py`

**Add your models name in** `user_config.py`

## Classifier

The classifier's `clarification_required` function will get an instruction and gridworld state as its inputs. Your class must implement this function and return the classication result. The output should 0 or 1 - 0 if clarification is not required, 1 if clarification is required. Invalid outputs may cause the submission to fail.

See `classifiers/random_classifier.py` for an example.

## Ranker

The ranker's `rank_questions` function will get an instruction, gridworld state, a list of clarifying questions and the max number of questions to be returned as its inputs. All instructions may not need a clarifying question, however this information is not provided to the ranker. Questions that do not need clarificaiton will not be used for scoring the ranker.  Your class must implement the `rank_questions` function and return the list of questions, such that the best matching question is closest to index 0. Note that returning a list longer than the max number specified has no penalty, but will questions above the limit will not be scored.

See `rankers/random_ranker.py` for an example.


## What's used by the evaluator

The evaluator uses `UserClassifer` and `UserRanker` from `user_config.py` as its entrypoints. 