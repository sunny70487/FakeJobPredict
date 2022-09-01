#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import math
from sklearn.preprocessing import normalize
#匯入資料
proc_df = pd.read_csv("./data/fake_job_postings.csv") 
#資料預處理
proc_df = proc_df.where(pd.notnull(proc_df), ' ')
proc_df['aggr_post'] = proc_df["company_profile"] + " " + proc_df["description"]+ " " + proc_df["requirements"]+ " " + proc_df["benefits"]



import langid

def detect_lang(x):
    code,_ = langid.classify(x)
    
    return code

proc_df = proc_df[proc_df['aggr_post'].apply(lambda x: detect_lang(x) == 'en')]

import re
import string

def clean_text(text):
    text = text.lower()                                              # make the text lowercase
    text = re.sub('\[.*?\]', '', text)                               # remove text in brackets
    text = re.sub('http?://\S+|www\.\S+', '', text)                  # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)                 # remove links
    text = re.sub('<.*?>+', '', text)                                # remove HTML stuff
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # get rid of punctuation
    text = re.sub('\n', '', text)                                    # remove line breaks
    text = re.sub('\w*\d\w*', '', text)                             # remove anything with numbers, if you want
    text = re.sub(r'[^\x00-\x7F]+',' ', text)                       # remove unicode
    return text

proc_df['aggr_post'] = proc_df['aggr_post'].apply(lambda x: clean_text(x))
proc_df.head()

corpus = proc_df['aggr_post'].values
#計算詞語出現頻率
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)  

#獲取詞袋中所有文本的關鍵字
word = vectorizer.get_feature_names()
#查看詞頻结果  
df_word =  pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())



#調用函數 
transformer = TfidfTransformer(smooth_idf=True,norm='l2',use_idf=True)  
#將計算好的詞頻矩陣統計為TF-IDF值  
tfidf = transformer.fit_transform(X)  
#查看計算的tf-idf
df_word_tfidf = pd.DataFrame(tfidf.toarray(),columns=vectorizer.get_feature_names())
#查看計算的idf
df_word_idf = pd.DataFrame(list(zip(vectorizer.get_feature_names(),transformer.idf_)),columns=['單詞','idf'])
df_word_idf.sort_values(by=['idf'],ascending=False,inplace=True)

vectorizer = CountVectorizer(vocabulary=df_word_idf['單詞'][:5000])
x = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()
df1 = pd.DataFrame(x.toarray(),columns=vectorizer.get_feature_names())

proc_df = proc_df.reset_index(drop = True)
df1 = df1.reset_index(drop = True)
df = proc_df.join(df1)
df.to_csv('TF_IDF.csv')




