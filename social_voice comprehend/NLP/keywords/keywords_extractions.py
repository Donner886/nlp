from __future__ import division

import pandas as pd
import re
import nltk
try:
    from nltk.stem import WordNetLemmatizer
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
import os
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize  # import figsize
from wordcloud import WordCloud
import collections
from NLP.sentiment.sentiment_classification import sentiment_classify
from collections import defaultdict

path = os.path.dirname(__file__)
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


stop_words = stopwords.words('english')
sw = set(stop_words[116:120] + stop_words[131:135] + stop_words[143:] + ['further', 'any', 'more'])
stop_words = list(set(stop_words) - set(sw)) + list('abcdefghijklmnopqrstuvwxyz') + ["I've", "I'm", "I'll", 'theyve']
stop_words = list(set([w.replace("'", "").lower() if "'" in w else w.lower() for w in stop_words]))


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
# word grouping
group_words = {
'good':['good', 'great', 'excellent', 'nice', 'awesome', 'best', 'not bad', 'amazing', 'perfect', 'ok', 'okay', 'wonderful','fantastic','fine', 'better','well', 'wxcellent'],       
'love':['love', 'loving', 'like','loved', 'liked'],
'difficult':['difficult', 'difficulty'],
'convenience':['convenient', 'convienience', 'conveniet', 'efficient', 'effective', 'efficiency', 'convinient', 'conveinent', 'convinenet', 'handy'],
'easy': ['easily','simple'],
'fast': ['quick','faster','quickly'],
'secure':['safe', 'secured' ],
'smooth': ['stable',],
'save': ['saves', 'saved'],
'clear': ['fresh'],
'app':['app!', 'aps', 'ap'],
'cant':['cannot', 'couldnt'],
'can':['could'],
'doesnt':['dont','didnt'],
'please':['pls', 'plz', 'plss', 'pleas', 'plea'],
'ui':['interface', 'design', 'layout'],
'user':['users'],
'fingerprint':['finger print'],
'would':['will'],
'create':['creating', 'creates', 'created'],
'come':['coming','comes', 'came'],
'access':['acces'],
'see':['sees','see', 'saw', 'seen'],
'note':['notes', 'noted'],
'use':['use', 'using', 'used', 'uses'],
'rely':['relies', 'relied'],
'remittance':['remittances'],
'satisfy': ['satisfied', 'satisfying'],
'improve':['improving','improved','improves'],
'confuse': ['confused', 'confusing', 'confuses'],
'rebate': ['rebates', 'rebating'],
'pay': ['paid', 'paying', 'payment'],
'minute':['min', 'mins', 'minutes'],
'thing':['things'],
'make': ['mades','made', 'made', 'making'],
'pass':['pas'],
'update':['updates', 'updated', 'updating']}

def group_W(word):
    flag = 0
    for key, mylist in group_words.items():
        if word in mylist:
            flag = 1
            return key
            break
    if flag == 0:
        return word

corpus_list_word = list(pd.read_csv(os.path.join(path,'corpus_list_word.txt'),header=None)[0].str.strip())
def get_change(comment):
    # input a string and output a string
    skip_set = ['updates', 'updated', 'updating', 'hing', 'ding', 'wed', 'sing', 'red', 'ios', 'pls','pleas', 'plea','ap','aps','dbs','fps', 'sms','pass','pas','pleased','mades','made', 'made', 'making', 'relies','min', 'thing', 'things', 'cos', 'mins', 'thing', 'minutes','satisfied','paying','rebates','rebating','improving','improved', 'improves','confused', 'confusing', 'confuses', 'created', 'relied','noted','remittance', 'use', 'using', 'used', 'uses', 'notes','sees','seeing', 'saves', 'acces', 'coming','comes','creating', 'creates', 'mas', 'banking', 'using', 'useless', 'always', 'access', 'news','dls', 'funding', 'fds']
    comment = comment.split(' ')
    change = []
    for w in comment:
        flag = 0
        if len(w)>=3 and w not in skip_set:
            if w[-3:] == 'ing':
                if w[:-3] in corpus_list_word:
                    w = w[:-3]
                    flag = 1
            elif w[-2:] == 'es':
                if w[:-2] in corpus_list_word:
                    w = w[:-2]
                    flag = 1
            elif w[-1:] == 's':
                if w[:-1] in corpus_list_word:
                    w = w[:-1]
                    flag = 1
            elif w[-2:] == 'ed':
                if w[:-2] in corpus_list_word:
                    w = w[:-2]
                    flag = 1
        if flag == 0 and group_W(w) != '':
            w = group_W(w)
        change.append(w)
    return ' '.join(change).strip()

# %%
filter_word = {
        'simplicity_or_convenience': ['complicate','confuse','conveniently','simply','simple','easier','convenience', 'easy', 'concise', 'difficult', 'complicated', 'inconvenient', 'inconvenience'],
        'time_consuming': ['consume','wait','timely','shortly','slowly','immediately','longer','endless','quicker', 'fast', 'faster' ,'fastest' 'speed','waste', 'long', 'slow', 'time', 'second', 'hour', 'minute'],
        'decvice_compatible': ['wei','hua','xiao','mi','samgsung','compatibility','iphonex','huawei','xiaomi','soni','apple','galaxy','ipad','htc','lg', 'nokia','samsung','device', 'iphone','phone','andriod','android', 'compatible', 'incompatible', 'os', 'ios', 'io'],
        'security':['secure', 'security','privacy', 'risk', 'safety'],
        'blank_wrong':['blank', 'wrong'],
        'biometric_identification': ['touch','touchid','recognition','recognize','recognise','authentication','biometric', 'fingerprint', 'face','faceid', 'facial', 'print', 'finger'],
        'OCR':['identify','authenticate','scanner','scan', 'scanning', 'scam', 'camera', 'picture', 'photo','id','hkid', 'identity', 'detect','detection','identification'],
        'QR_Code_method':['qr'],
        'clear':['clear', 'clean', 'unclear', 'clearly'],
        'language_display':['cn','chinese', 'english', 'language','eng'],
        'charge':['penalty','price','fee', 'charge', 'free', 'high', 'low', 'freely'],
        'size':['big','small', 'huge', 'large', 'little'],
        'app_performance': ['work', 'open', 'restart', 'terminate','kill','steadily','smooth', 'smoothly', 'bug', 'stable', 'server','system', 'error', 'crash','busy', 'stuck', 'buggy', 'steady', 'unstable'],
        'verification_otp':['verify','mail','messager','messenger','otp','msg','sms','sm', 'token', 'email', 'message', 'verification'],
        'area_use':['uk','globally','abroad', 'journey','travel','area', 'region','mainland','oversea', 'global', 'foreign', 'country'],
        
        
        'login_operation':['login', 'access', 'logon'],
        'loading':['load','reload'],
        'upload_operation':['upload','file','document'],
        'install_operation':['download', 'install', 'instal','reinstall', 'uninstall'],
        'update':['update', 'upgrade'],
        'banking':['bank', 'banking'],
        'transaction_way':['send','transfer','pay', 'transaction', 'fps','payee'],
        'money_management':['wealth','withdraw','cashless','coin','cheque','save', 'saving','money', 'bill', 'cash', 'limit', 'atm', 'branch', 'deposit', 'wallet'],
        'loan_management':['loan', 'debt', 'debit', 'morgage'],
        'fund_or_insurance_management':['fund', 'mpf', 'insurance'],
        'expense':['expense', 'budget'],
        'fx_service':['fx', 'forex', 'exchange', 'currency', 'currencies', 'hkd', 'dollar'],
        'rate_offer':['rate','interest'],
        'account_registration':['account', 'accout' 'register', 'registration'],
        'card_management':['card', 'credit'],
        'interface':['display','beautifully','ui','layout', 'interface', 'ugly'],
        'customer_service':['service', 'agent', 'hotline','answer', 'reply', 'chat', 'telephone'],
        'password_management':['password', 'pw', 'pin', 'passcode'],
        'function':['function', 'functional', 'feature', 'functionality'],
        'icon_operation':['key','click', 'page','screenshot', 'screen', 'button', 'keyboard', 'icon', 'capture'],
        'experience':['experience', 'ux'],
        'investment_service':['stock', 'trade', 'trading', 'invest', 'investment'],
        'internet_connection':['wifi','online', 'net', 'web', 'website', 'internet', 'disconnect','connect','connection', 'netword'],
        'notification': ['notice','notify','notification', 'note'],
        'promotion':['promotional','reward', 'promotion', 'coupon', 'discount', 'benefit'],
        'recordation':['record', 'history', 'logging', 'log', 'logs'],
        'other_feature':['confirmation', 'menu', 'friendly', 'unfriendly', 'address'],
        'advertisement':['ad', 'advertising', 'advertisement'],
        'shopping_service':['toabao', 'shop', 'shopping', 'alibaba', 'hktvmall'],
        'instruction':['instruction', 'statement', 'guide', 'guideline','reference'],
        }


def group_features(keyword, polarity = 'neutral'):
    word_label = []
    detail_word_label = []
    flag = 0
    for x in keyword.split(' '):
        for key, value in filter_word.items():
            if x in value and polarity == 'negative':
                word_label.append('bad_'+key)
                detail_word_label.append('bad_'+key+':'+keyword)
                flag = 1
                
                break
            elif x in value and polarity == 'positive':
                word_label.append('good_'+key)
                detail_word_label.append('good_'+key+':'+keyword)
                flag = 1
                break
    if flag == 0:
        word_label.append('others')
        detail_word_label.append('others'+':'+keyword)
    return list(set(word_label)),list(set(detail_word_label))


def get_groups(s):
    return group_features(s)[0]
def get_groups_detail(s):
    return group_features(s)[1]

# %%
# Extract keywords
def remove_punctuation(comment):
    return re.sub("[^a-zA-Z\u4e00-\u9fa5]", ' ', comment).strip().lower()


corpus_list = list(pd.read_csv(os.path.join(path,'corpus.txt'), header=None)[0].str.strip())

def get_keywords(s, max_words=3, stopwords=stop_words):
    '''
    1. return keywords if rake algorithm run successfully.
    2. return original sentence if other case.
    3. input 's' must be a series.
    '''
    s = str(s)
    language = classify_language(s)

    rake = Rake(max_words=max_words, language_code=language, stopwords=stopwords)
    if language == 'en':
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
        raise Exception("Invalid input: only English is supported so far.")
    return s_keyword


# word cloud
def union_keywords(text):
    text_set = set(text) - set(' ')
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
        text = union_keywords(keyword_list)

    elif language == 'zh':
        raise Exception("Invalid input: only English is supported so far.")
        # text = union_keywords(keyword_list)
        # text = [w for w in text if w not in sw]
    return text


# draw word cloud and return a phrase-frequency pair dictionary
def draw_wordcloud(text, polarity='neutral', background_color="white", max_words=200, max_font_size=130, random_state=1, width=1500,
                   height=700, dpi=1080, name='test'):
    phrase_counts = collections.Counter(text)
    phrase = []
    counts = []
    group_word = []
    group_word_detail = defaultdict(list)
    
    for key, value in phrase_counts.items():
        if key != '':
            phrase.append(key)
            counts.append(value)
            group_word_, group_word_detail_list = group_features(keyword = key, polarity=polarity)
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
    
    df_counts = pd.DataFrame({'keywords': phrase, 'frequency': counts})
    df_counts = df_counts.sort_values(by=['frequency'], ascending=False)

    wc = WordCloud(background_color=background_color,
                   max_words=max_words,
                   max_font_size=max_font_size,
                   random_state=random_state,
                   font_path=os.path.join(path,'msjh.ttc'),
                   width=width,
                   height=height).generate_from_frequencies(group_word_cloud)
    figsize(8, 6)
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi

    plt.imshow(wc)
    plt.axis("off")
    
    path2 = os.path.join(os.path.dirname(path),'result')
    if not os.path.exists(path2):
        os.makedirs(path2)

    plt.savefig(os.path.join(path2, name + ".jpeg"), dpi=1080)
    
    writer = pd.ExcelWriter(os.path.join(path2, "frequency_"+name+"_result.xlsx"))
    df_counts.to_excel(writer, index=False, sheet_name='keywords')
    df_group.to_excel(writer, index=False, sheet_name='group')
    writer.save()
    

    # plt.show()
    return df_counts, df_group


def frequency_wordcloud(keyword_list, name='test', polarity='neutral'):
    df_counts, df_group = draw_wordcloud(word_dictionary(keyword_list), name=name, polarity=polarity)
    return df_counts, df_group

def extract_tags(sentence, keyword):
    sentence, lem_list, tags = lemmatize_all(remove_info(data_clearing(sentence)), tag_result=True)
    word = []
    tag = []
    for i in range(len(lem_list)):
        if lem_list[i].strip() in ' '.join(keyword):
            word.append(lem_list[i].strip())
            tag.append(tags[i])
    return pd.DataFrame({'word':word, 'tag': tag})