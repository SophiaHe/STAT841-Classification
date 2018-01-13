# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:03:27 2017

@author: hyq92
"""

##############################################################################
# model: NB with all training data's 1st column & 3rd column: 0.67833 on kaggle
# Cannot run on jupyterhub because of memory constrain

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
df = pd.read_csv("train.csv", header=-1)
pd.DataFrame.head(df)
df.columns = ['0', '1', '2', '3']

df2 = df
df2['y'] = df2['0']
df2['QA'] = df2['1'] +  df2['3'] 
df2 = df2.drop(['0', '1', '3', '2'],axis=1)
pd.DataFrame.head(df2)

from sklearn.model_selection import train_test_split
df_train1, df_val1 = train_test_split(df2, train_size = 0.8,test_size=0.2)
print(pd.DataFrame.head(df_train1))
print(pd.DataFrame.head(df_val1))

""" Naive Bayes classifier """
bayes_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())
                      ])
bayes_clf.fit(df_train1["QA"].values.astype('U'), df_train1["y"])
# joblib.dump(bayes_clf, "bayes_20newsgroup.pkl", compress=9)
""" Predict the test dataset using Naive Bayes"""
predicted = bayes_clf.predict(df_val1["QA"].values.astype('U'))
print('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == df_val1["y"])))

df_test = pd.read_csv("test.csv", header=-1)

df_test.columns = ['0', '1', '2']
df_test = pd.DataFrame(df_test.iloc[1:,:])
df_test['QA'] = df_test['0'] + df_test['2']
df_test = df_test.drop(['0', '1', '2'],axis=1)


pd.DataFrame.head(df_test)
len(df_test)

bayes_clf1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())
                      ])
bayes_clf1.fit(df2["QA"].values.astype('U'), df2["y"])

predicted_test = bayes_clf1.predict(df_test["QA"].values.astype('U'))
# np.savetxt("nb_1and3rd_col.csv", predicted_test, delimiter=",") # 0.67833 on kaggle

##############################################################################
# SVM trained on entire dataset's 1st & 3rd column: 0.63022 on kaggle
df2 = df
df2['y'] = df2['0']
df2['QA'] = df2['1'] +  df2['3'] 
df2 = df2.drop(['0', '1', '3', '2'],axis=1)
pd.DataFrame.head(df2)


""" Support Vector Machine (SVM) classifier"""
svm_clf = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=   5, random_state=42)),
])
svm_clf.fit(df2['QA'].values.astype('U'), df2['y'])

df_test = pd.read_csv("test.csv", header=-1)

df_test.columns = ['0', '1', '2']
df_test = pd.DataFrame(df_test.iloc[1:,:])
df_test['QA'] = df_test['0'] + df_test['2']
df_test = df_test.drop(['0', '1', '2'],axis=1)

predicted_test = svm_clf.predict(df_test["QA"].values.astype('U'))
# np.savetxt("svm_1and3rd_col_alldata.csv", predicted_test, delimiter=",") # 0.63022 on kaggle

##############################################################################
# Best model: NB with all training data's 1st & 3rd column + stop word + 
#        FitPrior=False: 0.68333 on kaggle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd
import numpy as np


df2 = df
df2['y'] = df2['0']
df2['QA'] = df2['1'] +  df2['3'] 
df2 = df2.drop(['0', '1', '3', '2'],axis=1)
pd.DataFrame.head(df2)

from sklearn.model_selection import train_test_split
df_train1, df_val1 = train_test_split(df2, train_size = 0.8,test_size=0.2)
print(pd.DataFrame.head(df_train1))
print(pd.DataFrame.head(df_val1))





bayes_clf2 = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(fit_prior =False)),])
bayes_clf2.fit(df_train1["QA"].values.astype('U'), df_train1["y"])

predicted = bayes_clf2.predict(df_val1["QA"].values.astype('U'))
print('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == df_val1["y"])))

# use entire training set to train & predict on df_test
bayes_clf2 = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(fit_prior =False)),])
bayes_clf2.fit(df2["QA"].values.astype('U'), df2["y"])

df_test = pd.read_csv("test.csv", header=-1)

df_test.columns = ['0', '1', '2']
df_test = pd.DataFrame(df_test.iloc[1:,:])
df_test['QA'] = df_test['0'] + df_test['2']
df_test = df_test.drop(['0', '1', '2'],axis=1)


predicted_test = bayes_clf2.predict(df_test["QA"].values.astype('U'))
# np.savetxt("nb_1and3rd_col2.csv", predicted_test, delimiter=",") # 0.68333  on kaggle

##############################################################################
#model: NB with all training data's 1st & 3rd column + stop word + 
#       FitPrior=False + stemming: 0.67922 on kaggle
import nltk
#nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                             ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False)),])

    
text_mnb_stemmed = text_mnb_stemmed.fit(df_train1["QA"].values.astype('U'), df_train1["y"])
predicted_mnb_stemmed = text_mnb_stemmed.predict(df_val1["QA"].values.astype('U'))
np.mean(predicted_mnb_stemmed == df_val1["y"]) # 0.67665714285714285

# train on all training data
text_mnb_stemmed1 = text_mnb_stemmed.fit(df2["QA"].values.astype('U'), df2["y"])
predicted_mnb_stemmed1 = text_mnb_stemmed1.predict(df_test["QA"].values.astype('U'))
# np.savetxt("nb_1and3rd_col3.csv", predicted_mnb_stemmed1, delimiter=",") # 0.67922  on kaggle

############################################################################
# ensemble method: 0.68338 on kaggle, which is only a very small improvement compared to previous submission: 0.68333
from collections import Counter
def majority_element(a):
    c = Counter(a)
    value, count = c.most_common()[0]
    if count > 1:
        return value
    else:
        return a[0]
# read in predictions from different classifiers 
nb_1and3rd_col = pd.read_csv("ensemble method predictions/nb_1and3rd_col.csv", header=-1)
nb_1and3rd_col2 = pd.read_csv("ensemble method predictions/nb_1and3rd_col2.csv", header=-1)
nb_1and3rd_col3 = pd.read_csv("ensemble method predictions/nb_1and3rd_col3.csv", header=-1)
svm_1and3rd_col_alldata = pd.read_csv("ensemble method predictions/svm_1and3rd_col_alldata.csv", header=-1)

merged_predictions = [[s[0],s[1],s[2],s[3]] for s in zip(nb_1and3rd_col[1], nb_1and3rd_col2[1], nb_1and3rd_col3[1], svm_1and3rd_col_alldata[1])]
majority_prediction = [majority_element(p) for p in merged_predictions]

