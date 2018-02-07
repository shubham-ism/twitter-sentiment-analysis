from nltk import sent_tokenize,word_tokenize
#C:\Users\Shubham Gupta\PycharmProjects\sentdex\venv\lib\site-packages\nltk\__init__.py
#C:\Users\Shubham Gupta\AppData\Roaming\nltk_data\corpora\state_union
example_text = "This is awesome. Roronoa Zoro is world's greatest swordsman. Luffy is gonna be the king of pirates."

#print(word_tokenize(example_text))
#print(sent_tokenize(example_text))

'''Stock Word - a, an, the '''
////////////////////////////////////

'''Stop Word Filteration'''
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
print(stop_words)

example_text = "Luffy is gonna be the king of pirates. Roronoa Zoro is gonna be the world's greatest swordsman. Sanji is gonna find the all blue. Chopper is gonna be a pirate. Nami is gonna draw the map of the whole world."

temp = []
for i in word_tokenize(example_text):
    if(i not in stop_words):
        temp.append(i)

print(temp)

/////////////////////////////////
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example_words = ["python","pythoning","pythoner","pythoned","pythonly"]

#for w in example_words:
#    print(ps.stem(w))

example_text = "It is very important to be pythonly wuth python. At least once every pythoner has pythoned poorly"
words = word_tokenize(example_text)

for w in words :
    print(ps.stem(w))



////////////////////////////////////////////////////////
'''Part of Speech Tagging'''
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(train_text)
def process_content():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        print(tagged)

process_content()


/////////////////////////////////////////////////////////////
'''Chunking using parser'''
'''Part of Speech Tagging'''
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(train_text)
def process_content():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        # print(tagged)
        Chunk = r"""chunk: { <NN>} """
        chunkPaser = nltk.RegexpParser(Chunk)
        chunked = chunkPaser.parse(tagged)
        print(chunked)


process_content()
 ////////////////////////////////////////////////////////
#chunking and chinking
'''Part of Speech Tagging'''
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(train_text)
def process_content():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        # print(tagged)
        Chunk = r"""chunk: { <NN>} """
        chunkPaser = nltk.RegexpParser(Chunk)
        chunked = chunkPaser.parse(tagged)
        #print(chunked)
        chunked.draw()


process_content()
////////////////////////////////////////////////////////
#Named Entity Recognition
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized = custom_sent_tokenizer.tokenize(train_text)
def process_content():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        namedEnt = nltk.ne_chunk(tagged, binary=True)
        namedEnt.draw()


process_content()
///////////////////////////////////////////////
#lemmanting
'''Kinda like stemming but gives an actual word'''
from nltk.stem import WordNetLemmatizer

Lammetizer = WordNetLemmatizer()

print(Lammetizer.lemmatize("cats"))
print(Lammetizer.lemmatize("better"))   #default pos in a noun
print(Lammetizer.lemmatize("better", pos = "a"))   #if you want a differnt result sent in the position {a means adjective}
print(Lammetizer.lemmatize("run","v"))
# Better tham stemming
////////////////////////////////////////////////
#find location
import nltk

print(nltk.__file__) #to get the location of nltk or any other stuff on python

////////////////////////////////////////////////
#corpora
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample_text = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample_text)

print(tok[5:15])

///////////////////////////////////////////////
#WordNet
#synsets and similarity
from nltk.corpus import wordnet
syns = wordnet.synsets("program")


#synset
#print(syns)
#print(syns[0])
#print((syns[0].name()))
print(syns[0].lemmas())
print(syns[0].lemmas()[0])

#just the word
print(syns[0].lemmas()[0].name())


#definition
print(syns[0].definition())

#Examples
print(syns[0].examples())


synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if( l.antonyms() ) :
            antonyms.append(l.antonyms()[0].name())


print(set(synonyms))
print(set(antonyms))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cactus.n.01")
print(w1.wup_similarity(w2))

////////////////////////////////////////////////
## Text Classifier
# Just Two Choices
# You Can also create your own list (2 choices)

import nltk
import  random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
'''
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category) :
    documents.append((list(movie_reviews.words(fileid)), category))


>>> nltk.corpus.gutenberg.fileids()             #to get the meaning of fileid
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']
'''
random.shuffle(documents)

#print(documents[0])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))


print(all_words["mind blowing"])

//////////////////////////////////////////////////////////
# Text Classifier
# Just Two Choices
# You Can also create your own list (2 choices)

import nltk
import  random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
'''
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category) :
    documents.append((list(movie_reviews.words(fileid)), category))


>>> nltk.corpus.gutenberg.fileids()             #to get the meaning of fileid
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']
'''

random.shuffle(documents)

#print(documents[0])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["mind blowing"])


////////////////////////////////////////////////////////////
#
# Text Classifier
# Just Two Choices
# You Can also create your own list (2 choices)

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
'''
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category) :
 //   documents.append((list(movie_reviews.words(fileid)), category))


>>> nltk.corpus.gutenberg.fileids()             #to get the meaning of fileid
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']
'''

random.shuffle(documents)

# print(documents[0])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["mind blowing"])


word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in words_features:
        features[w] = (w in words)

    return features


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]

'''
featuresets contains (dictionary,category ) tuple
dictionary is 'word_name' : 'True'|'False'
'''
//////////////////////////////////////////////////////////////////////////////////
#
# Text Classifier
# Just Two Choices
# You Can also create your own list (2 choices)

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
'''
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category) :
 //   documents.append((list(movie_reviews.words(fileid)), category))


>>> nltk.corpus.gutenberg.fileids()             #to get the meaning of fileid
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
'shakespeare-macbeth.txt', 'whitman-leaves.txt']
'''

random.shuffle(documents)

# print(documents[0])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["mind blowing"])


word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [((find_features(rev)), category) for (rev, category) in documents]
# naive base algorithm
'''
In top 3000 words these are the words thats were there or not in this movie

'''
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occurences x liklihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

////////////////////////////////////////////////////////////////
#pickle
#save python object
import nltk
import  random
from nltk.corpus import movie_reviews
import pickle
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def  find_features (document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

'''
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
'''
////////////////////////////////////////////////////////////////////
#scikit learn preimpleted algorithm (Like Naive Baise)
##scikit learn
#marry things
#Scikit classifier

import nltk
import  random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def  find_features (document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# GaussianNB, BernoulliNB
'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print(" GaussianNB_classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
LogisticRegression, SGDClassifier
SVC, LinearSVC, NuSVC
'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)
'''
warning appears because skikit learn new version has added new parameters and because we have not added that value 
its taking default as 1000 
you can stop the warning by adding value = 1000 but so that the code works with previous versions
its best not to add it 
'''

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


////////////////////////////////////////////////////////////////////////////////
#voting system on classifier

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# GaussianNB, BernoulliNB
'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print(" GaussianNB_classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
LogisticRegression, SGDClassifier
SVC, LinearSVC, NuSVC
'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)
'''

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassiifer(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0]))
'''
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0]))
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0]))
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0]))
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0]))
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0]))


'''

//////////////////////////////////////////////////////////////////////////////
#vote
#voting system on classifier

import nltk
import  random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI  # to import from classify
from statistics import mode



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





documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def  find_features (document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# GaussianNB, BernoulliNB
'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print(" GaussianNB_classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
LogisticRegression, SGDClassifier
SVC, LinearSVC, NuSVC
'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)
'''

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassiifer(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)









////////////////////////////////////////////////////////////////////////////////
#investigating Bais
#Investigating Bias
import nltk
import  random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI  # to import from classify
from statistics import mode



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

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def  find_features (document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]

#positive data
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#negative data
training_set = featuresets[100:]
testing_set = featuresets[:100]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# GaussianNB, BernoulliNB
'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print(" GaussianNB_classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
LogisticRegression, SGDClassifier
SVC, LinearSVC, NuSVC
'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)
'''

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassiifer(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
'''
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
'''


///////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#Better training data
import nltk
import  random
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

'''
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
'''

short_pos = open("short_reviews/positive.text","r").read()
short_neg = open("short_reviews/negative.text","r").read()

documents = []

for r in short_pos.split('\n')
    documents.append( (r, "pos") )

for r in short_neg.split('\n')
    documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def  find_features (document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]

random.shuffle(featuresets)

#positive data
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

'''
#negative data
training_set = featuresets[100:]
testing_set = featuresets[:100]
'''

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)


# GaussianNB, BernoulliNB
'''
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print(" GaussianNB_classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
LogisticRegression, SGDClassifier
SVC, LinearSVC, NuSVC
'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)
'''

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassiifer(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
'''
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
'''

////////////////////////////////////////////////////////////////

#
#Better training data
import nltk
import  random
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

'''
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
'''

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:2000]

def  find_features (document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]

random.shuffle(featuresets)

#positive data
training_set = featuresets[:3000]
testing_set = featuresets[3000:6000]

'''
#negative data
training_set = featuresets[100:]
testing_set = featuresets[:100]
'''

classifier = nltk.NaiveBayesClassifier.train(training_set)

'''
classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''


print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassiifer(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
'''
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
'''
////////////////////////////////////////////////////////////////////////////////////
#pickle everything
import nltk
import  random
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


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

save_documents = open("documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:4000]

save_word_features = open("wordfeatures.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def  find_features (document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [ ((find_features(rev)), category) for (rev, category) in documents]

save_featuresets = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

training_set = featuresets[:3000]
testing_set = featuresets[3000:6000]

'''
#negative data
training_set = featuresets[100:]
testing_set = featuresets[:100]
'''

classifier = nltk.NaiveBayesClassifier.train(training_set)

'''
classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''


save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("mnb.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("Bernoullinb.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("logisticregression.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

save_classifier = open("sgdclassifier.pickle", "wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

save_classifier = open("linearsvcclassifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

save_classifier = open("nusvcclassifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassiifer(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
'''
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
'''
////////////////////////////////////////////////////////////////////////////////////
#use pickled data
import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
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
        #print(votes)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf



documents_f = open("documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

'''
word_features_f = open("wordfeatures.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
'''

featuresets_f = open("featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

'''
training_set = featuresets[:3000]
'''
testing_set = featuresets[3000:6000]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#MNB classifier
MNB_classifier_f = open("mnb.pickle","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#Bernoulli classifier
BernoulliNB_classifier_f = open("Bernoullinb.pickle","rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
BernoulliNB_classifier_f.close()
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

#LogisticRegression_classifier
LogisticRegression_classifier_f = open("logisticregression.pickle","rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()
print(" LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#SGDClassifier_classifier
SGDClassifier_classifier_f = open("sgdclassifier.pickle","rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
SGDClassifier_classifier_f.close()
print(" SGDClassifier_classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

#LinearSVC_classifier
LinearSVC_classifier_f = open("linearsvcclassifier.pickle","rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

#NuSVC_classifier
NuSVC_classifier_f = open("nusvcclassifier.pickle","rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

#voted classifier
voted_classifier = VoteClassiifer(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

#classification result in percentage
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)

//////////////////////////////////////////////////
#pickle everything
# pickle everything and make module
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


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

# move this up here
documents = []
all_words = []

# j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]
for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)  # pos conatains tuple of ("word", "TAG")
    for w in pos:
        if w[1][0] in allowed_word_types:  # w[1][0]    1 refers to TAG and 0 refers to the first letter
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())  # add that word in all_words

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [((find_features(rev)), category) for (rev, category) in documents]

save_featuresets = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(word_features, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:3000]
testing_set = featuresets[3000:6000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

save_classifier = open("pickled_algos/originalnaivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print(" BernoulliNB_classifier accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print(" LogisticRegression_classifier accuracy percent: ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print(" SGDC_classifier accuracy percent: ", (nltk.classify.accuracy(SGDC_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

'''
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

save_classifier = open("pickled_algos/NuSVC_classifier5k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()
'''

voted_classifier = VoteClassiifer(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier, )

print(" voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)


def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats)

//////////////////////////////////////////////////////////////////////////////////
#convert it into a module


# pickle everything and make module
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

calssifier_f = open("pickled_algos/originalnaivebayes.pickle","rb")
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



def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


#SAVED AS sentiment_mod.py


///////////////////////////////////////////////
#run sentiment on a input
import sentiment_mod as s

print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was all all. Horrible movie 0/10"))



//////////////////////////////////////////////////////////////////////////////
#sentiment_mod
# pickle everything and make module
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

calssifier_f = open("pickled_algos/originalnaivebayes.pickle","rb")
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



def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


#SAVED AS sentiment_mod.py

/////////////////////////////////////////
#twitter sentiment analysis with output file in twitter-out.txt
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="OHSWiLZWApbtfERhITz6XQOik"
csecret="sbDcIsjBKZ0JHQvZBGZDOeOFy8Jp7wn8EpuvTkWHW7zN21neaQ"
atoken="960085083519684608-DIKmwQPzW20xrHGp8kahBy0MPASeqbS"
asecret="MDTgFXsqIMRq54d7XjIR5gopBMQJ92vPrmsGPQkbwEX4z"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet,sentiment_value, confidence*100)

        if(confidence*100 >= 80):
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["modi"])



///////////////////////////////////////////////////////////////////////
#plot graph
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("twitter-out.txt","r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines:                 #[-200:]
        x += 1
        if "pos" in l:
            y += 1
        elif "neg " in lines:
            y -= 1

        xar.append(x)
        yar.append(y)

    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
















