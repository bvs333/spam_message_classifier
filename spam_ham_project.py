# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:41:17 2019

@author: bvsre
"""

import pandas as pd

spam_ham_dataset = pd.read_json("data1.json")

import nltk

nltk.download('stopwords') 

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import re

corpus=[]
for i in range(5572):
    message = re.sub('[^a-zA-Z]', ' ', spam_ham_dataset['v2'][i])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus.append(message)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

spam_ham_features = cv.fit_transform(corpus).toarray()
spam_ham_labels = spam_ham_dataset.iloc[:,3].values


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(spam_ham_features, spam_ham_labels, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)


labels_pred = classifier.predict(features_test)

classifier.score(features_test,labels_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(labels_test, labels_pred)