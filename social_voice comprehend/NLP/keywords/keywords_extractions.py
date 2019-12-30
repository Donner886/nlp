from __future__ import division

import pandas as pd
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
from NLP.keywords.algorithm import Rake
from NLP.keywords.group_word import group_features_en, group_features_zh, get_change
import os
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize  # import figsize
from wordcloud import WordCloud
import collections
from NLP.sentiment.sentiment_classification import sentiment_classify
from collections import defaultdict
import jieba
from NLP.opencc import OpenCC

path = os.path.dirname(__file__)
jieba.load_userdict(os.path.join(path,"F_Cantonese_dictionary2.txt"))

def data_clearing(comment):
    # remove punctuation and extract words
    comment = str(comment)
    comment = re.sub('[\\\’\']', "", comment)
    comment = re.sub("[^a-zA-Z\u4e00-\u9fa5\.\,\!\?\，\。\？\！\“\”]", ' ', comment).strip().lower()
    return comment

def classify_language(text):
    # Label language category, remove punctuations and integrate sentences by different language
    r = re.compile(u'[\u4e00-\u9fa5]+')
    if re.search(r, str(text)):
        return ('zh')
    else:
        return ('en')

def s2hk_word(comment):
    cc = OpenCC('s2hk')
    return cc.convert(comment)

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
    elif 'Hey' in comment.split(' '):
        comment = comment.split('Hey')[0]
    elif 'hey' in comment.split(' '):
        comment = comment.split('hey')[0]

    # remove url
    if 'https:' in comment or 'www.' in comment:
        url_reg = r'[a-z]*[:.]+\S+'
        comment = re.sub(url_reg, '', comment)
    return comment


# Lemmatization
def lemmatize_all(sentence, tag_result=False):
    wnl = WordNetLemmatizer()
    lem_list = []
    tags = []
    flag = 1
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            lem_list.append(' ' + wnl.lemmatize(word, pos='n'))
            tags.append('NN')
            flag = 0
        elif tag.startswith('VB'):
            lem_list.append(' ' + wnl.lemmatize(word, pos='v'))
            tags.append('VB')
            flag = 0
        elif tag.startswith('JJ'):
            lem_list.append(' ' + wnl.lemmatize(word, pos='a'))
            tags.append('JJ')
            flag = 0
        elif tag.startswith('R'):
            lem_list.append(' ' + wnl.lemmatize(word, pos='r'))
            tags.append('R')
            flag = 0
        elif (tag == '.' or tag == ',') and flag == 0:
            lem_list.append(wnl.lemmatize(word, pos='r'))
            tags.append('.')
            flag = 1
        elif (tag == '.' or tag == ',') and flag == 1:
            pass
        else:
            lem_list.append(' ' + word)
            tags.append('others')
            flag = 0
    if tag_result:
        return ''.join(lem_list).strip(), lem_list, tags
    else:
        return ''.join(lem_list).strip()

# English stopwords
stop_words = stopwords.words('english')
sw = set(stop_words[116:120] + stop_words[131:135] + stop_words[143:] + ['further', 'any', 'more'])
stop_words = list(set(stop_words) - set(sw)) + list('abcdefghijklmnopqrstuvwxyz') + ["I've", "I'm", "I'll", 'theyve']
stop_words = list(set([w.replace("'", "").lower() if "'" in w else w.lower() for w in stop_words]))

# Chinese stopwords
stop_words_zh=[line.rstrip('\n').strip() for line in open(os.path.join(path, 'F_total_tradition_stopwords.txt'), 'r', encoding='utf-8-sig')]

def remove_stopwords_lemma(comment):
    # lemmatisation
    c = lemmatize_all(comment)                      # split the text into words and lemmatize them --> list
    c_list = [w for w in c if w not in stop_words]  # remove stop words --> list
    comment = ' '.join(c_list)
    return comment


# remove scores
def remove_scores(w_list):
    if w_list != []:
        w_list = [x[0] for x in w_list]
    return w_list


# remove single words and subset
def remove_singleWord_subset(wordlist, language='en'):
    if type(wordlist) == list:
        if language == 'en':
            wordlist = [str(w) for w in wordlist if len(str(w).split(' ')) > 1]
        else:
            wordlist = [w for w in wordlist if len(w) > 1]
        wordlist = [w for w in wordlist if w not in ' '.join(list(set(wordlist) - set([w])))]
    return wordlist


# %%
# Extract keywords
def remove_punctuation(comment):
    return re.sub("[^a-zA-Z\u4e00-\u9fa5]", ' ', comment).strip().lower()


corpus_list = list(pd.read_csv(os.path.join(path,'corpus.txt'), header=None)[0].str.strip())
corpus_list_zh = list(pd.read_csv(os.path.join(path,'corpus_zh.txt'), header=None)[0].str.strip())

def get_keywords(s, max_words=3):
    '''
    1. return keywords if rake algorithm run successfully.
    2. return original sentence if other case.
    3. input 's' must be a series.
    '''
    s = str(s)
    language = classify_language(s)
    if language == 'en':
        rake = Rake(max_words=max_words, language_code=language, stopwords=stop_words)
        s = get_change(lemmatize_all(remove_info(data_clearing(s))))
        s_T = s
        s_keyword1 = remove_scores(rake.apply(s_T))

        # use the whole corpus as input and execute matching

        s_T = re.sub("[\.\,\!\?]", ' ', s_T).split(' ')
        s_T = [x.strip() for x in s_T if x not in stop_words + ['', ' ']]
        s_T = ' '.join(s_T)
        s_keyword2 = [w for w in corpus_list if str(w) in s_T]  # matching

        s_keyword1 = [remove_punctuation(str(w)) for w in s_keyword1]
        # s_keyword2 = [remove_punctuation(str(w)) for w in s_keyword2]
        word = list(set(s_keyword1) | set(s_keyword2))  # | set(s_keyword3[i]))
        word = remove_singleWord_subset(word, language)
        if len(word) > 0:
            s_keyword = list(set(word))
        else:
            t0 = s.split(' ')
            t0 = [remove_punctuation(str(w)) for w in t0]
            t0 = [w for w in t0 if w not in stop_words + ['', ' ']]
            s_keyword = list(set(t0))
            
    else:
        rake = Rake(max_words=max_words, language_code=language, stopwords=stop_words_zh)
        # single comment
        s = s2hk_word(remove_info(data_clearing(s)))
        s_T = s
        s_keyword1 = rake.apply(s_T)
        
        # whole corpus
        s_T = re.sub("[\.\,\!\?\，\。\？\！\“\”]", ' ', s_T)
        s_T = jieba.lcut(s_T)
        s_T = [x.strip() for x in s_T if x not in stop_words_zh+['', ' ']]
        s_T = ' '.join(s_T)
        s_keyword2 = [w for w in corpus_list_zh if w in s_T]  # matching
    
        # Union keywords from single comment input and whole corpus comment input
        s_keyword = []
        
        word = list(set(s_keyword1) | set(s_keyword2))
        word = remove_singleWord_subset(word, language)
        if len(word) > 0:
            s_keyword=list(set(word))
        else:
            t0 = jieba.lcut(s)
            t0 = [remove_punctuation(str(w)) for w in t0]
            t0 = [w for w in t0 if w not in stop_words_zh+['', ' ']] 
            s_keyword=list(set(t0))
    return s_keyword


# word cloud
def union_keywords(text, language='en'):
    text_set = set(text) - set(' ')
    if language == 'en':
        for t in list(text_set):
            substract_list = list(text_set - set([t]))
            for t_y in substract_list:
                if len(set(t.split(' ')) & set(t_y.split(' '))) == len(set(t.split(' '))) and len(set(t.split(' '))) > 1:
                    while t in text:
                        text.remove(t)
                        text.append(t_y)
    text = [t for t in text if t!='others']
    return text


# generate candidate phrases
def word_dictionary(keyword_list):
    if type(keyword_list[0]) == list:
        keyword_list = [word for mylist in keyword_list for word in mylist]
    language = classify_language(' '.join(keyword_list))
    if language == 'en':
        text = union_keywords(keyword_list, language=language)
        text = [w for w in text if w not in [' ']]

    elif language == 'zh':
        #raise Exception("Invalid input: only English is supported so far.")
        text = union_keywords(keyword_list, language=language)
        text = [w for w in text if w not in stop_words_zh]
    return text


# draw word cloud and return a phrase-frequency pair dictionary
def draw_wordcloud(text, output=None, showgroup=False, polarity='neutral', background_color="white", max_words=200, max_font_size=130, random_state=1, width=1500,
                   height=700, dpi=1080, name='test'):
    #text_zh = [w for w in text if classify_language(w) == 'zh']
    text_zh =  [w for w in text if classify_language(w) == 'zh']
    text_en =  [w for w in text if classify_language(w) == 'en']
    #text_en = re.sub("[a-zA-Z]", ' ', '|'.join(text)).split('|')
    #text_zh = re.sub("[\u4e00-\u9fa5]", ' ', '|'.join(text)).split('|')
    #language = classify_language(' '.join(text))
    phrase = []
    counts = []
    group_word_cloud = {}
    group_word_cloud_zh = {}
    word_draw = {}
    df_group = pd.DataFrame()
    df_group_zh = pd.DataFrame()
    if len(text_en)>0:
        phrase_counts = collections.Counter(text_en)
        group_word = []
        group_word_detail = defaultdict(list)
        
        for key, value in phrase_counts.items():
            if key != '':
                phrase.append(key)
                counts.append(value)
        for key in text_en:
            group_word_, group_word_detail_list = group_features_en(keyword = key, polarity=polarity)
            group_word.extend(group_word_)
            for k in group_word_detail_list:
                group_word_detail[k.split(':')[0]].append(k.split(':')[1])
        group_word_cloud = [w for w in group_word if w != 'others']
        group_word_cloud = collections.Counter(group_word_cloud)
        group_counts = collections.Counter(group_word)
        group_counts = sorted(group_counts.items(), key=lambda x:x[1], reverse=True)
        df_group = pd.DataFrame(group_counts, columns=['group', 'freq'])
        
        group_detail_freq = []
        for group_key in df_group['group']:
            detail_counts = collections.Counter(group_word_detail[group_key])
            detail_counts = sorted(detail_counts.items(), key=lambda x:x[1], reverse=True)
            group_detail_freq.append(detail_counts)
        df_group['group_detail'] = group_detail_freq
        word_draw.update(group_word_cloud)
        
    if len(text_zh)>0:
        phrase_counts = collections.Counter(text_zh)
        group_word = []
        group_word_detail = defaultdict(list)
        
        for key, value in phrase_counts.items():
            if key != '':
                phrase.append(key)
                counts.append(value)
        for key in text_zh:
            group_word_, group_word_detail_list = group_features_zh(keyword = key, polarity=polarity)
            group_word.extend(group_word_)
            for k in group_word_detail_list:
                group_word_detail[k.split(':')[0]].append(k.split(':')[1])
        group_word_cloud_zh = [w for w in group_word if len(w)>1 and w not in stop_words_zh+['每次','垃圾','银行','对着','工作','平时','早就','最终','确实','方式','事情','本来','也许','这样的话','昨天','姐妹','上次','不用','允悲','心情','有人','前提','从来不','反正','感觉','others','东西','不会','可爱','普及','随意','下午','每次','马化腾','地方','马云','媳妇','为啥','没','原因','理由','游泳','鹿晗','关晓彤','只不过','偷偷']]
        group_word_cloud_zh = collections.Counter(group_word_cloud_zh)
        group_word_cloud_zh = {k:v for k, v in group_word_cloud_zh.items() if v>1}
        
        group_counts = sorted(group_word_cloud_zh.items(), key=lambda x:x[1], reverse=True)
        
        df_group_zh = pd.DataFrame(group_counts, columns=['group', 'freq'])
        
        group_detail_freq = []
        for group_key in df_group_zh['group']:
            detail_counts = collections.Counter(group_word_detail[group_key])
            detail_counts = sorted(detail_counts.items(), key=lambda x:x[1], reverse=True)
            group_detail_freq.append(detail_counts)
        df_group_zh['group_detail'] = group_detail_freq
        word_draw.update(group_word_cloud_zh)

    df_group = pd.concat([df_group, df_group_zh])
    df_counts = pd.DataFrame({'keywords': phrase, 'frequency': counts})
    df_counts = df_counts.sort_values(by=['frequency'], ascending=False)

    wc = WordCloud(background_color=background_color,
                   max_words=max_words,
                   max_font_size=max_font_size,
                   random_state=random_state,
                   font_path=os.path.join(path,'msjh.ttc'),
                   width=width,
                   height=height).generate_from_frequencies(word_draw)
    figsize(8, 6)
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi

    plt.imshow(wc)
    plt.axis("off")

    plt.savefig(os.path.join(output, name + ".jpeg"), dpi=1080)
    
    #writer = pd.ExcelWriter(os.path.join(path2, "frequency_"+name+"_result.xlsx"))
    #df_counts.to_excel(writer, index=False, sheet_name='keywords')
    #df_group.to_excel(writer, index=False, sheet_name='group')
    #writer.save()
    

    # plt.show()
    if not showgroup:
        return df_counts
    else:
        return df_counts, df_group


def frequency_wordcloud(keyword_list, name='test', polarity='neutral', output='', showgroup=False):
    if showgroup:
        df_counts, df_group = draw_wordcloud(word_dictionary(keyword_list), name=name, output=output, polarity=polarity, showgroup=showgroup)
        return df_counts, df_group
    else:
        df_counts = draw_wordcloud(word_dictionary(keyword_list), name=name, output=output, polarity=polarity, showgroup=showgroup)
        return df_counts

def extract_tags(sentence, keyword):
    sentence, lem_list, tags = lemmatize_all(remove_info(data_clearing(sentence)), tag_result=True)
    word = []
    tag = []
    for i in range(len(lem_list)):
        if lem_list[i].strip() in ' '.join(keyword):
            word.append(lem_list[i].strip())
            tag.append(tags[i])
    return pd.DataFrame({'word':word, 'tag': tag})