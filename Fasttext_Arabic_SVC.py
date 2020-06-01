from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import csv
import pandas as pd
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import FastText
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
# define training data
##sample = open('COVID19TweetsLabeled.txt', "r",encoding= 'utf-8') 
##s = sample.read() 
##
### Replaces escape character with space 
##f = s.replace("\n", " ") 
##  
##data1 = [] 
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
data=[]
data_labels=[]
with open('LabelTweets2000.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=',')
    line_count = 0
    for row in reader:    
        data.append(row[3])
        data_labels.append(row[7])
        line_count += 1
    print(f'Processed {line_count} lines.')

csvFile.close()
### iterate through each sentence in the file 
##for i in sent_tokenize(f): 
##    temp = [] 
##      
##    # tokenize the sentence into words 
##    for j in word_tokenize(i): 
##        temp.append(j.lower()) 
##  
##    data1.append(temp)
sentences = data
common_texts=sentences
# train model
#model = Word2Vec(sentences, min_count=1)
model = FastText(size=8, window=6, min_count=1)  # instantiate
model.build_vocab(sentences=common_texts)
model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
print(len(common_texts))
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


Pipe_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("classifier", SVC())])
X_train, X_test, y_train, y_test = train_test_split(data,data_labels, test_size=0.25, random_state=33)

classifier = Pipe_w2v
classifier.fit(X_train,y_train)
# predict
predictions = classifier.predict(X_test)

c=accuracy_score(y_test,predictions)
print('Accurcy of using Logistic Regression')
print(c)

from sklearn.metrics import f1_score
print('F1 of using Logistic Regression')
f1 = f1_score(y_test,predictions, average='macro')
print(f1)
from sklearn.metrics import recall_score
print('recall of using Logistic Regression')
r1 = recall_score(y_test,predictions, average='macro')
print(r1)
from sklearn.metrics import precision_score
print('precision of using Logistic Regression')
p1 = precision_score(y_test,predictions, average='macro')
print(p1)


##Pipe_w2v2= Pipeline([
##    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
##    ("classifier", LinearSVC())])
##X_train, X_test, y_train, y_test = train_test_split(data,data_labels, test_size=0.25, random_state=33)
##
##classifier2 = Pipe_w2v2
##classifier2.fit(X_train,y_train)
### predict
##predictions2 = classifier2.predict(X_test)
##
##c=accuracy_score(y_test,predictions2)
##print('Accurcy of using LinearSVC')
##print(c)
##print('done')
##
##Pipe_w2v = Pipeline([
##    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
##    ("classifier", LogisticRegression())])
##X_train, X_test, y_train, y_test = train_test_split(data,data_labels, test_size=0.25, random_state=33)
##
##classifier = Pipe_w2v
##classifier.fit(X_train,y_train)
### predict
##predictions = classifier.predict(X_test)
##
##c=accuracy_score(y_test,predictions)
##print('Accurcy of using Logistic Regression')
##print(c)


##Pipe_w2v2= Pipeline([
##    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
##    ("classifier", LinearSVC())])
##X_train, X_test, y_train, y_test = train_test_split(data,data_labels, test_size=0.25, random_state=33)
##
##classifier2 = Pipe_w2v2
##classifier2.fit(X_train,y_train)
### predict
##predictions2 = classifier2.predict(X_test)
##
##c=accuracy_score(y_test,predictions2)
##print('Accurcy of using LinearSVC')
##print(c)
##print('done')

