import json
import os,ast
import io, re,string
import csv
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
import csv
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from stop_words import safe_get_stop_words
from sklearn.metrics import accuracy_score
#stop_words = safe_get_stop_words('arabic')
###################################
from os import path
import codecs
import nltk
import arabic_reshaper
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#set(stopwords.words('arabic'))

def removeWeirdChars(text):
    weridPatterns = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)
    return weridPatterns.sub(r'', text)
#from bidi.algorithm import get_display

stop_words = []
##print(stop_words)
d = path.dirname(__file__)
f = codecs.open(path.join(d, 'list.txt'), 'r', 'utf-8')
text = removeWeirdChars(f.read())
##print(text)
word_tokens = word_tokenize(text)
for w in word_tokens: 
        stop_words.append(w)

stop_words.append('كورونا')
stop_words.append('فيروس')
stop_words.append('بفيروس')
stop_words.append('كرونا')
stop_words.append('19')
stop_words.append('كوفيد')
stop_words.append('فايروس')


    
#########################read csv file#############################

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
vectorizer =  CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 1),
    stop_words=stop_words,
    min_df = 2,
    #max_df=200000
)
X= vectorizer.fit_transform(data)
Y=data_labels
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=33)
    
classifer = LogisticRegression()
classifer.fit(X_train, Y_train)
scores = cross_val_score(classifer,X_test, Y_test, cv=10)
print (classifer)
print (scores)
# predict
predictions = classifer.predict(X_test)

c=accuracy_score(Y_test,predictions)
print('Accurcy of using Logistic Regression')
print(c)
from sklearn.metrics import f1_score
print('F1 of using Logistic Regression')
f1 = f1_score(Y_test,predictions, average='macro')
print(f1)
from sklearn.metrics import recall_score
print('recall of using Logistic Regression')
r1 = recall_score(Y_test,predictions, average='macro')
print(r1)
from sklearn.metrics import precision_score
print('precision of using Logistic Regression')
p1 = precision_score(Y_test,predictions, average='macro')
print(p1)
print('#############################################################################################')
X_train, X_test, y_train, y_test = train_test_split(data,data_labels, test_size=0.25, random_state=33)     
vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,1))
X_train_m = vectorizer.fit_transform(X_train)
X_test_m= vectorizer.transform(X_test)

y_train_m= vectorizer.transform(y_train)
y_test_m = vectorizer.transform(y_test)
print(X_train_m.shape)
print(y_train_m.shape)
print("Dimensions of train data X:",X_train_m.shape, "Y :",y_train_m.shape)
print("Dimensions of test data X:",X_test_m.shape,"Y:",y_test_m.shape)

classifier = LogisticRegression()

# train
classifier.fit(X_train_m, y_train)

# predict
predictions = classifier.predict(X_test_m)

c=accuracy_score(y_test,predictions)
print('Accurcy of using Logistic Regression')
print(c)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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

print('done')	
