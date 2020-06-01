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

with open('COVID19Tweets.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=',')
    line_count = 0
    for row in reader:    
        data.append(row[3])
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


true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++')#, max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :30]:
        print(' %s' % terms[ind]),
    print

print("\n")
clust_labels = model.predict(X)
cent = model.cluster_centers_
print(clust_labels)

##################################################
##from sklearn.metrics.pairwise import cosine_similarity
##dist = 1 - cosine_similarity(X)
##import os  # for os.path.basename
##
##
##import matplotlib.pyplot as plt
##import matplotlib as mpl
##
##from sklearn.manifold import MDS
##
##MDS()
##
### convert two components as we're plotting points in a two-dimensional plane
### "precomputed" because we provide a distance matrix
### we will also specify `random_state` so the plot is reproducible.
##mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
##
##pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
##
##xs, ys = pos[:, 0], pos[:, 1]
##clusters = model.labels_.tolist()
##df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))#, title=titles))
##cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
##
###set up cluster names using a dict
##cluster_names = {0: 'Family, home, war', 
##                 1: 'Police, killed, murders', 
##                 2: 'Father, New York, brothers', 
##                 3: 'Dance, singing, love', 
##                 4: 'Killed, soldiers, captain'}
##
###group by cluster
##groups = df.groupby('label')
##
##
### set up plot
##fig, ax = plt.subplots(figsize=(17, 9)) # set size
##ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
##
###iterate through groups to layer the plot
###note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
##for name, group in groups:
##    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
##            label=cluster_names[name], color=cluster_colors[name], 
##            mec='none')
##    ax.set_aspect('auto')
##    ax.tick_params(\
##        axis= 'x',          # changes apply to the x-axis
##        which='both',      # both major and minor ticks are affected
##        bottom='off',      # ticks along the bottom edge are off
##        top='off',         # ticks along the top edge are off
##        labelbottom='off')
##    ax.tick_params(\
##        axis= 'y',         # changes apply to the y-axis
##        which='both',      # both major and minor ticks are affected
##        left='off',      # ticks along the bottom edge are off
##        top='off',         # ticks along the top edge are off
##        labelleft='off')
##    
##ax.legend(numpoints=1)  #show legend with only 1 point
##
###add label in x,y position with the label as the film title
###for i in range(len(df)):
###    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  
##
##    
##    
##plt.show() #show the plot
##
##print()
##print('done')

print('done')	
