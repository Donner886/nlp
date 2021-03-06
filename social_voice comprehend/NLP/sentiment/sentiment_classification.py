from __future__ import division

import warnings
import re
import nltk
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    wnl = WordNetLemmatizer()
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import joblib
import jieba
import os
path = os.path.dirname(__file__)
warnings.filterwarnings('ignore')

vectorizer = joblib.load(os.path.join(path,"vectorizer.pkl"))
model = joblib.load(os.path.join(path,"model.pkl"))

vectorizer_zh = joblib.load(os.path.join(path,"vectorizer_zh.pkl"))
ch2_zh = joblib.load(os.path.join(path,"ch2_zh.pkl"))
model_zh = joblib.load(os.path.join(path,"model_zh.pkl"))

# data preprocessing
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

# %% English
def remove_sw(comment):
    stop_words = stopwords.words('english')
    sw = set(
        stop_words[116:120] + stop_words[131:135] + stop_words[143:] + ['but', 'how', 'just', 'why', 'further', 'very',
                                                                        'any', 'more', 'if'])
    stop_words = list(set(stop_words) - set(sw)) + list('abcdefghijklmnopqrstuvwxyz') + ["I've", "I'm", "I'll",
                                                                                         'theyve']
    stop_words = list(set([w.replace("'", "").lower() if "'" in w else w.lower() for w in stop_words]))

    clist = word_tokenize(comment)
    clist = [w for w in clist if w not in stop_words]
    return ' '.join(clist)


# Lemmatization
def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    lem_list = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            lem_list.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lem_list.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lem_list.append(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            lem_list.append(wnl.lemmatize(word, pos='r'))
        else:
            lem_list.append(word)
    return ' '.join(lem_list)

# %% Chinese
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

def pre_processing(comment, lan):
    if lan == 'en':
        return lemmatize_all(remove_sw(remove_info(comment)))
    else:
        return chinese_segmentation(remove_info(comment))


def sentiment_classify(comment):
    comment_T = data_clearing(comment)
    lan = classify_language(comment_T)
    X_test0 = [pre_processing(str(comment), lan)]
    
    if lan == 'en':
        X_test = vectorizer.transform(X_test0)
        result = int(model.predict(X_test))
    else:
        X_test = vectorizer_zh.transform(X_test0)
        X_test = ch2_zh.transform(X_test)
        result = int(model_zh.predict(X_test))
        
    if result == 1:
        return ('positive')
    else:
        return ('negative')
#    else:
#        raise Exception("Invalid input: input must be a string.")

## usage
# sentiment_classify('inMotion is a very good app! It makes life easier')
# sentiment_classify('The app always crashes and I will not use it again')
