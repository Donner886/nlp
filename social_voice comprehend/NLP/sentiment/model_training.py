from __future__ import division

import pandas as pd
import numpy as np
import os 
import warnings
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import collections
from collections import defaultdict
import jieba
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from NLP.opencc import OpenCC
from sklearn.externals import joblib

path = os.path.dirname(__file__)
warnings.filterwarnings('ignore')

# %%
def data_clearing(comment):
    # remove punctuation and extract words
    comment = str(comment)
    comment = re.sub('[\\\’\']', "", comment)
    comment = re.sub("[^a-zA-Z\u4e00-\u9fa5]", ' ', comment).strip().lower()
    return comment


def classify_language(text):
    # Label language category, remove punctuations and integrate sentences by different language
    r = re.compile(u'[\u4e00-\u9fa5]+')
    if re.search(r, str(text)):
        return ('zh')
    else:
        return ('en')


# remove unnecessary information
def remove_info(comment):
    # remove replied information
    if 'the hongkong and shanghai banking corporation' in comment:
        comment = comment.split('the hongkong and shanghai banking corporation')[0]
    elif 'standard chartered bank' in comment:
        comment = comment.split('standard chartered bank')[0]
    elif 'BankMobile a Division of Customers Bank' in comment:
        comment = comment.split('BankMobile a Division of Customers Bank')[0]
    elif 'Citibank (Hong Kong) Limited' in comment:
        comment = comment.split('Citibank (Hong Kong) Limited')[0]

    if '开发者回复' in comment:
        comment = comment.split('开发者回复')[0]
    elif 'Hi' in comment.split('Hi'):
        comment = comment.split('Hi')[0]
    elif 'Hey' in comment.split('Hey'):
        comment = comment.split('Hey')[0]
    elif 'hi' in comment.split('hi'):
        comment = comment.split('hi')[0]
    elif 'hey' in comment.split('hey'):
        comment = comment.split('hey')[0]

    # remove url
    if 'https:' in comment or 'www.' in comment:
        url_reg = r'[a-z]*[:.]+\S+'
        comment = re.sub(url_reg, '', comment)
    return comment

# import chinese stopwords
sw_S = [line.rstrip('\n').strip() for line in open(os.path.join(path, 'F_total_tradition_stopwords_sentiment.txt'), 'r', encoding='utf-8-sig')]

# update cantonese dictionary in jieba
jieba.load_userdict(os.path.join(path, 'F_Cantonese_dictionary2.txt'))

# Chinese segmentation
def chinese_segmentation(comment):
    comment = jieba.lcut(comment)
    comment = [w.strip() for w in comment if w.strip() not in sw_S]
    comment = ' '.join(comment)
    return comment
# %% 

# Vecterization
def weight_method(X_train0 = None, X_test0 = None, y_train=None, weight='tf-idf',max_features=1000, max_selected = 100):
    if weight == 'one-hot':
        # one-hot encoding
        vectorizer = CountVectorizer(binary=True, max_features=max_features, min_df=0, token_pattern='\w+')
        vectorizer.fit(X_train0)
        X_train = vectorizer.transform(X_train0)
        X_test = vectorizer.transform(X_test0)
    if weight == 'n-grams':
        # n-grams
        vectorizer = CountVectorizer(binary=True, max_features=max_features, ngram_range=(1, 3), min_df=0, token_pattern='\w+')
        vectorizer.fit(X_train0)
        X_train = vectorizer.transform(X_train0)
        X_test = vectorizer.transform(X_test0)
    if weight == 'count':
        # word counts
        vectorizer = CountVectorizer(binary=False, max_features=max_features, min_df=0, token_pattern='\w+')
        vectorizer.fit(X_train0)
        X_train = vectorizer.transform(X_train0)
        X_test = vectorizer.transform(X_test0)
    if weight == 'tf-idf':
        # tf-idf
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=0, token_pattern='\w+')
        vectorizer.fit(X_train0)
        X_train = vectorizer.transform(X_train0)
        X_test = vectorizer.transform(X_test0)
    
    ch2 = SelectKBest(chi2, k=max_selected)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test, vectorizer, ch2

# Return the best model
def findparameter(classifier, tuned_parameters, X_train, X_test, y_train, y_test, name):    
    if tuned_parameters is not None:
        # Classify the data using xxClassifier
        gsearch1 = GridSearchCV(estimator = classifier, param_grid = tuned_parameters, 
                                cv=10, scoring='f1') #cv=5
        gsearch1.fit(X_train,y_train)

        para = gsearch1.best_params_
        
        # Perform hyperparameter tuning / model selection
        model = gsearch1.best_estimator_
        
    else:
        para = None
        model = classifier
        model.fit(X_train, y_train)

    # Return best classifier model
    predictions  = model.predict(X_test)
    accuracy     = accuracy_score(y_test,predictions)
    recallscore  = recall_score(y_test,predictions)
    precision    = precision_score(y_test,predictions)
    roc_auc      = roc_auc_score(y_test,predictions)
    f1score      = f1_score(y_test,predictions) 
    kappa_metric = cohen_kappa_score(y_test,predictions)
    predictions_train  = model.predict(X_train)
    accuracy_train     = accuracy_score(y_train,predictions_train)
    
    df = pd.DataFrame({"Model"           : [name],
                       "training acc"    : [accuracy_train],
                       "testing acc"     : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve": [roc_auc],
                       "Kappa_metric"    : [kappa_metric],
                       "Para"            : [para['C']],
                      })
    
    print(name, ' -> ', 'training acc: ', accuracy_train, ' ||  testing acc: ', accuracy)
    
    return model,df, para

# Plot confusion metrics
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def s2hk_word(comment):
    cc = OpenCC('s2hk')
    return cc.convert(comment)

def pre_processing(comment):
    comment_T = data_clearing(comment)
    lan = classify_language(comment_T)
    if lan == 'en':
        return lemmatize_all(remove_sw(remove_info(comment_T)))
    else:
        return chinese_segmentation(s2hk_word(remove_info(comment_T)))


# %%
#  training model
df['comment_T'] = df['comment'].map(pre_processing)
df=df[~df['comment_T'].isin([''])] 
df = df.reset_index(drop=True)   

data = df[['comment_T','re_rank2']]
x = list(data['comment_T'])
y = data['re_rank2']

X_train0, X_test0, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3, stratify=y)

# %%
weight_ = ['tf-idf', 'n-grams', 'one-hot']
paras = defaultdict(list)

auc_score = 0
for i in range(len(weight_)):
    for n in range(500, 6000, 500):
        for m in range(100, n+200,200):
            weight = weight_[i]#'tf-idf' # n-grams， count， one-hot
            X_train, X_test, vectorizer, ch2 = weight_method(X_train0, X_test0, y_train, weight=weight, max_features=n, max_selected=m)    
            svm = SVC(random_state=1, kernel = 'linear')
            tuned_parameters = [{'C': np.linspace(0.8,1.8,10)}] #0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1
            clf_svm, metric3, para3 = findparameter(svm, tuned_parameters, X_train, X_test, y_train, y_test, 'SVM')
            paras['svm'].append(para3)
            
            if metric3['Area_under_curve'].iloc[0] > auc_score:
                joblib.dump(vectorizer, os.path.join(path,'vectorizer.pkl'))
                joblib.dump(clf_svm, os.path.join(path,'model.pkl'))
                joblib.dump(ch2, os.path.join(path,'ch2.pkl'))
            
            if i == 0 and n==100 and m==100:
                model_performances = pd.concat([metric3],axis = 0).reset_index(drop=True)
                #model_performances = model_performances.drop(columns = "index",axis =1)
                model_performances['weight'] = weight
                model_performances['max_features'] = n
            else:
                model_performances0 = pd.concat([metric3],axis = 0).reset_index()
                model_performances0 = model_performances0.drop(columns = "index",axis =1)
                model_performances0['weight'] = weight
                model_performances0['max_features'] = n
                model_performances = pd.concat([model_performances,model_performances0],axis = 0).reset_index(drop=True)
            print('>>>>>>',weight)
        
        
        
        
        # Perceptron
        #    ppn = Perceptron(tol=1e-3)
        #    tuned_parameters = [{'max_iter': range(1000,3000,10),
        #                        'eta0': [0.1, 0.25, 0.5,0.75, 1]}]
        #    clf_ppn, metric1, para1 = findparameter(ppn, tuned_parameters, X_train, X_test, y_train, y_test, 'Perceptron')
        #    
            # Logistic Regression
    #        lr = LogisticRegression(multi_class='ovr')
    #        tuned_parameters = [{'C': np.linspace(0.1,1,10),
    #                            'penalty':['l1','l2']}]
    #        clf_lr, metric2, para2 = findparameter(lr, tuned_parameters, X_train, X_test, y_train, y_test, 'Logistic Regression')
    #        paras['lr'].append(para2)
            
            # SVM
            
    
