The package can support  sentences comprehend and extract the partitions of it.
1, sentiment of the sentences
2, keywords which can summarize major content
3, groups



============== Use Case =========================
from NLP.sentiment.sentiment_classification import sentiment_classify
from NLP.keywords.keywords_extractions import get_keywords, frequency_wordcloud, lemmatize_all, extract_tags, remove_info, data_clearing

# sentiment analysis
sentiment_classify('inMotion is a very good app! It makes life easier')
sentiment_classify('The app always crashes and I will not use it again')

# get keywords
get_keywords('inMotion is a very good app! It makes life easier')
get_keywords('The app always crashes and I will not use it again')

#keyword_list = get_keywords('inMotion is a very good app! It makes life easier')
#keyword_list.extend(get_keywords('The app always crashes and I will not use it again'))
#df_counts = frequency_wordcloud(keyword_list)


# test a data frame    
df_en['sentiment'] = df_en['comment'].map(sentiment_classify)
df_en['keywords'] = df_en['comment'].map(get_keywords)

# return group excel and wordcloud image
df_keywords_inMotion, df_group_inMotion = frequency_wordcloud(
        keyword_list = list(df_en[(df_en['sentiment']=='negative')&(df_en['bank']=='inMotion')]['keywords']), 
        name='negative_label_inMotion', polarity='negative')

df_keywords_inMotion_p, df_group_inMotion_p = frequency_wordcloud(
        keyword_list = list(df_en[(df_en['sentiment']=='positive')&(df_en['bank']=='inMotion')]['keywords']), 
        name='positive_label_inMotion', polarity='positive')
