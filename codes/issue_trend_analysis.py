#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
import os
import json
import re
import string
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import MWETokenizer
from sklearn.decomposition import TruncatedSVD

import gensim
import time
import warnings
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

import numpy as np
np.random.seed(222)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_and_split_to_tokens(sentences):
    """
    :param sentences: (array_like) array_like objects of strings.
        e.g., ["I like apples", "I love python3"]
    You can choose the level of pre-processing by yourself.
    The easiest way to start is lowering the case (str.lower).
    :return: array_like objects of array_like objects of tokens.
        e.g., [["I", "like", "apples"], ["I", "love", "python3"]]
    """
    punct_mapping = {"_":" ", "'":" "}
    punct = "/-'?!,#$%\'()*+-/:;<=>@\\^_`{|}~[]" + '""“”’'
    remove = ["[weekender]", "[newsmaker]", "[graphic news]", "(photo news)", "[herald interview]"]
    def clean_special_chars(text, punct, mapping):
        for r in remove:
            text = text.replace(r, "")
        for p in punct:
            text = text.replace(p, f' {p} ')
        for m in mapping:
            text = text.replace(m, mapping[m])
        
        return text

    modified_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        modified_sentence = clean_special_chars(sentence,punct,punct_mapping)
        modified_sentences.append(modified_sentence)
        
    return modified_sentences


def make_topictable_per_doc(ldamodel, corpus, texts):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.
        
        if len(doc) == 0:
            topic_table = topic_table.append(pd.Series([-1]), ignore_index=True)
        
        # 모든 문서에 대해서 각각 아래를 수행
        
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num)]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.

            else:
                break
    return(topic_table)


path = '../Data/'
number_of_topic = 125

datapaths = os.listdir(path)
datapaths.sort()

df = pd.DataFrame()
for p in datapaths:
    with open(path + p, 'r') as f:
        data = json.load(f)

    dataframe = pd.DataFrame.from_dict(data)
    df = df.append(dataframe)


text = pd.DataFrame()
text = df[['title', ' body', ' time']]
text[' body'] = preprocess_and_split_to_tokens(text[' body'])
text['title'] = preprocess_and_split_to_tokens(text['title'])

# Tokenization
text[' body'] = text.apply(lambda row: word_tokenize(row[' body']), axis=1)
text['title'] = text.apply(lambda row: word_tokenize(row['title']), axis=1)

# Remove stop words
stop = nltk.corpus.stopwords.words('english')
text[' body'] = text[' body'].apply(lambda x: [word for word in x if word not in (stop)])
text['title'] = text['title'].apply(lambda x: [word for word in x if word not in (stop)])

text[' body'] = text[' body'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
text['title'] = text['title'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])

text[' body'] = text[' body'].apply(lambda x: [word for word in x if len(word) > 3])
text['title'] = text['title'].apply(lambda x: [word for word in x if len(word) > 3])

dictionary = corpora.Dictionary(text[' body'])
corpus = [dictionary.doc2bow(text) for text in text[' body']]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = number_of_topic, id2word=dictionary, iterations= 400, passes=40)

cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
coherence = cm.get_coherence()
print("Coherence",coherence)
print('Perplexity: ', ldamodel.log_perplexity(corpus),'\n\n')

topictable = make_topictable_per_doc(ldamodel, corpus, text[' body'])
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['Doc_num', 'Issue']


#Save CSV Data
csv_data = pd.read_csv("../CSV_Data/ner_tagged.csv")
text = text.reset_index()

text['Doc_num'] = topictable['Doc_num']
text['Issue'] = topictable['Issue']
csv_data['Issue'] = topictable['Issue']

text.to_csv("../CSV_Data/preprocessed_data.csv")
csv_data.to_csv("../CSV_Data/_issue.csv")





