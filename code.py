'''
Read Data
'''
import json_lines
X = []
yz = []
with open('data.txt', 'rb') as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        yz.append([item['voted_up'], item['early_access']])

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Translate to English
from google_trans_new import google_translator
t = google_translator()
X_trans = []
for i in range(len(X)):
    print(f'{i}/{len(X)}')
    row = X[i][0]
    X_trans.append(t.translate(row))

# Split data
X_train, X_test, yz_train, yz_test = train_test_split(X_trans, yz, test_size=0.3)

yz_train_split = np.hsplit(np.array(yz_train), 2)
y_train = yz_train_split[0]
z_train = yz_train_split[1]

yz_test_split = np.hsplit(np.array(yz_test), 2)
y_test = yz_test_split[0]
z_test = yz_test_split[1]

'''
Feature Extraction
'''

import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

def preprocess(text_data):
    PUNCTUATION = '!"#$%&\'*+,-./;<=>?@[\\]^_`{|}~'

    def remove_punc(s):
        return "".join([char for char in s if char not in PUNCTUATION])

    def remove_stopwords(tokens):
        stop_words = stopwords.words('english')
        stop_words.remove('very')
        stop_words.remove('not')
        return [word for word in tokens if word not in stop_words]

    def stem(tokens):
        porter = PorterStemmer()
        return [porter.stem(word) for word in tokens]
    
    def encode_emojis(s):
        s = re.sub(r'♥+', 'profanity', s)
        s = s.replace(':)', 'smiley')
        s = s.replace(':(', 'frowney')
        s = s.replace('<3', 'heart')
        return s

    low = list(map(str.lower, text_data))
    emojis = list(map(encode_emojis, low))
    punc = list(map(remove_punc, emojis))
    tok = list(map(word_tokenize, punc))
    stop = list(map(remove_stopwords, tok))
    stemmed = list(map(stem, stop))

    # Represent X_stemmed as a single list of documents as strings with space separated tokens
    stemmed_flat = [' '.join(e for e in item) for item in stemmed]
    return stemmed_flat

# Word Count features
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=3500)
X_train_prep = preprocess(X_train)
X_train_count = vectorizer.fit_transform(X_train_prep)
X_test_prep = preprocess(X_test)
X_test_count = vectorizer.transform(X_test_prep)

# TF-IDF unigram features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3500)
X_train_prep = preprocess(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_prep)
X_test_prep = preprocess(X_test)
X_test_tfidf = tfidf.transform(X_test_prep)

# TF-IDF unigram & bigram features
X_train_prep = preprocess(X_train)
tfidf_bigram = TfidfVectorizer(max_features=3500, ngram_range=(1,2))
X_train_tfidf_bigram = tfidf_bigram.fit_transform(X_train_prep)
X_test_prep = preprocess(X_test)
X_test_tfidf_bigram = tfidf_bigram.transform(X_test_prep)

'''
Model Experiments & Evaluation
-> For each model a classification report was generated for predictions on both
   the train and test set. The same code was used each time, replacing the y parameter 
   in .fit(X, y) with values from {y_train, y_test, z_train, z_test} and the X parameter
   in .predict(X) with X_train_<F> or X_test_<F> where <F> is some feature in
   {count, tfidf, tfidf_bigram}.
'''

# Dummy Baseline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score

dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train_count, z_train)
dummy_y_pred = dummy_clf.predict(X_test_count)
print(classification_report(z_test, dummy_y_pred))

# Linear Regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Hyper-parameter selection

# C = [0.0001, 0.001, 0.01, 0.1, 1]
C = [1, 2, 3, 5, 10]
accuracy = []
std = []

for c in C:
    model = LogisticRegression(max_iter=250, C = c)
    scores = cross_val_score(model, X_train_tfidf, z_train, cv=5, scoring='accuracy')
    accuracy.append(scores.mean())
    std.append(scores.std())
    
plt.errorbar(C, accuracy, yerr=std)
plt.xlabel('c')
plt.ylabel('Accuracy')
plt.gcf().set_size_inches(10, 6)
plt.show()

# Logistic regression evaluation

lr_clf = LogisticRegression(max_iter=250, C=100)
lr_clf.fit(X_train_tfidf, z_train)
lr_y_pred = lr_clf.predict(X_test_tfidf)
print(classification_report(z_test, lr_y_pred))

# SVM Classifier

# Parameter tuning using GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parameter_space = {
    'C': [0.1, 1, 100, 1000],
    'gamma': [0.001, 0.1, 1, 3]
}

gsc = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=parameter_space, cv=5, scoring='accuracy')
grid_result = gsc.fit(X_train_tfidf, z_train)
best_params = grid_result.best_params_
print(best_params)

# SVM Evaluation
svm_clf = SVC(kernel='rbf', C=1000, gamma=0.001)
svm_clf.fit(X_train_tfidf_bigram, z_train)
svm_y_pred = svm_clf.predict(X_test_tfidf_bigram)
print(classification_report(z_test, svm_y_pred))

# Deep Neural Network
from sklearn.neural_network import MLPClassifier

# Parameter Selection
mlp_clf = MLPClassifier(max_iter=300)

parameter_space = {
    'hidden_layer_sizes': [(50, 30, 20), (50, 30, 20, 10), (30, 30, 20, 10, 10)],
    'alpha': [0.1, 0.0001],
}

grid_search = GridSearchCV(mlp_clf, parameter_space, cv=5, scoring='accuracy')
grid_search.fit(X_train_count, z_train)
mlp_best_params = grid_search.best_params_
print(f'Best params: {mlp_best_params}')

# Model Evaluation
mlp_clf = MLPClassifier(max_iter=300, hidden_layer_sizes=(50, 30, 20), alpha=0.0001)
mlp_clf.fit(X_train_count, z_train)
y_pred = mlp_clf.predict(X_test_count)
print(classification_report(z_test, y_pred))

'''
Learning Curves
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Accuracy")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times (s)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Performance of the model")

    return plt

fig, axes = plt.subplots(3, 3, figsize=(20, 15))

lr_clf = LogisticRegression(max_iter=250, C=1)
title = r"Learning Curves (Logistic Regression, C=1)"
plot_learning_curve(lr_clf, title, X_train_tfidf, y_train, axes=axes[:, 0],
                    cv=5, n_jobs=4)

svm_clf = SVC(kernel='rbf', C=1, gamma=1)
title = r"Learning Curves (SVM, C=1, $\gamma=1$)"
plot_learning_curve(svm_clf, title, X_train_tfidf_bigram, y_train, axes=axes[:, 1],
                    cv=5, n_jobs=4)

mlp_clf = MLPClassifier(max_iter=300, hidden_layer_sizes=(30, 30, 20, 10, 10), alpha=0.1)
title = r"Learning Curves (DNN, layers=(30,30,20,10,10), $\alpha=0.1$)"
plot_learning_curve(mlp_clf, title, X_train_count, y_train, axes=axes[:, 2],
                    cv=5, n_jobs=4)