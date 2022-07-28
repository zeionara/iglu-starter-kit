# from models.classifiers.random_classifier import RandomClassifier
from models.classifiers.bert_classifier import BERTClassifier
from models.rankers.random_ranker import RandomRanker

# UserClassifer = RandomClassifier
UserClassifer = BERTClassifier
UserRanker = RandomRanker