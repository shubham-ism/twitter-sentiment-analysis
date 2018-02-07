# pickle everything and make module
import os
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI  # to import from classify
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassiifer(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("pickled_algos/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k.pickle","rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets_f = open("pickled_algos/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:3000]
testing_set = featuresets[3000:6000]

calssifier_f = open("pickled_algos/originalnaivebayes5k.pickle","rb")
classifier = pickle.load(calssifier_f)
calssifier_f.close()

MNB_classifier_f = open("pickled_algos/MNB_classifier5k.pickle","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()

BernoulliNB_classifier_f = open("pickled_algos/BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
BernoulliNB_classifier_f.close()

LogisticRegression_classifier_f = open("pickled_algos/LogisticRegression_classifier5k.pickle","rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()

SGDClassifier_classifier_f = open("pickled_algos/SGDC_classifier5k.pickle","rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
SGDClassifier_classifier_f.close()

LinearSVC_classifier_f = open("pickled_algos/LinearSVC_classifier5k.pickle","rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
'''
NuSVC_classifier_f = open("pickled_algos/NuSVC_classifier5k.pickle","rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()
'''

voted_classifier = VoteClassiifer(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier, )

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)


def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


#SAVED AS sentiment_mod.py