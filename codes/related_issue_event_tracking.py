#!/usr/bin/env python
# coding: utf-8

# # Related_Issue_event

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim import corpora

import numpy as np
from collections import Counter
np.random.seed(222)

number_of_topic = 125
target_issue = 2
datapath = '../Data/ner_tagged.csv'
ssue_list = [0,1,2,3,4,5,6,7,8]
event_list = [0,1,2,3,4,5]

event_title_list = []



def lemma_NER(ner_list):    
    punc = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    punc += "“”’"
    ner_list = [word.translate(word.maketrans(punc, " "*len(punc))).strip() for word in ner_list]
    
    except_set = ['(yonhap)', 'yonhap', '', 'the', 'i', '000', 'south korea',
                  'we', 'north korea', 'kim', 'park', 'us', 'yonhap the']
    ner_list = [x for x in ner_list if x not in except_set]
    
    return ner_list

def extractNER(df):
    person = list(df['Person'])
    organization = list(df['Organization'])
    place = list(df['Geographical Entity'])

    person_list = []
    organization_list = []
    place_list = []

    for x in person:
        if type(x) == str:
            person_list += x.split(',')
    
    for x in organization:
        if type(x) == str:
            organization_list += x.split(',')
    
    for x in place:
        if type(x) == str:
            place_list += x.split(',')
    
    person_list = lemma_NER(person_list)
    organization_list = lemma_NER(organization_list)
    place_list = lemma_NER(place_list)

    person = Counter(person_list).most_common(1)[0][0]
    organization = Counter(organization_list).most_common(1)[0][0]
    place = Counter(place_list).most_common(1)[0][0]

    return person, organization, place


def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()



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




def choose_representative(temp_frame):
    
    max_score = 0
    max_score_sentence_index = -1
    body_list = []
    
    
    for index in range(len(temp_frame)):
        body_list.append((index, " ".join(temp_frame.iloc[index][ ' body'])))
    
    if len(body_list) == 1:
        return temp_frame.loc[0]['title']
    for index_n_body in body_list:
        index, body = index_n_body
        score = 0
        if not body:
            continue
        for index_n_comp_body in body_list:
            comp_index, comp_body = index_n_comp_body
            if not comp_body:
                continue
            if index!=comp_index:
                cosine_sim = get_cosine_sim(body, comp_body)[0][1]
                score += cosine_sim
        if len(body_list) != 1:  
            score /= (len(body_list) - 1)
        if score > max_score:
            max_score = score
            max_score_sentence_index = index
    return temp_frame.loc[max_score_sentence_index]['title']


def find_release_topic(issue_num, num_topic, list_text_topic, tokenized_doc, title_doc, title_target):
    new_tokenized_doc = []
    for i in list_text_topic[issue_num]:
        new_tokenized_doc.append(tokenized_doc[i])
    
    new_title_doc = []
    for i in list_text_topic[issue_num]:
        new_title_doc.append(title_doc[i])
          
    dictionary = corpora.Dictionary(new_tokenized_doc)
    corpus = [dictionary.doc2bow(text) for text in new_tokenized_doc]
    
    if any(corpus):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topic, id2word=dictionary, passes=30)
        topictable_issue = make_topictable_per_doc(ldamodel, corpus, new_tokenized_doc)
    else:
        topictable_issue = pd.DataFrame()
        for i in range(len(list_text_topic[issue_num])):
            topictable_issue = topictable_issue.append(pd.Series([0]), ignore_index=True)
    
    topictable_issue.columns = ['Issue']
    topictable_issue['Doc_num'] = pd.Series(list_text_topic[issue_num])
    topictable_issue[' body'] = pd.Series(new_tokenized_doc)
    topictable_issue['title'] = pd.Series(new_title_doc)

    if any(corpus) and (title_target == issue_num):
        for event in range(num_topic):
            temp_frame = pd.DataFrame()
            for i in range(len(topictable_issue)):
                if topictable_issue.iloc[i]["Issue"] == event:
                    temp_frame = temp_frame.append(topictable_issue.iloc[i], ignore_index=True)
            event_title_list.append(choose_representative(temp_frame))

    return topictable_issue



topictable = pd.read_csv("../Data/preprocessed_data.csv")

import ast 
topictable['title'] = topictable['title'].apply(ast.literal_eval)
topictable[' body'] = topictable[' body'].apply(ast.literal_eval)

list_text_topic = [[] for i in range(number_of_topic)]
for index in range(len(topictable)):
    list_text_topic[int(topictable.loc[index, 'Issue'])].append(index)


eventtable = []
for i in range(number_of_topic):
    if len(list_text_topic[i]) == 0:
        eventtable.append([])
        continue
    eventtable.append(find_release_topic(i, 6, list_text_topic, topictable[' body'], topictable['title'], target_issue))



event_index = pd.DataFrame()

for i in range(number_of_topic):
    if len(eventtable[i]) == 0:
        continue
    event_index = event_index.append(eventtable[i])

event_index.sort_values('Doc_num', inplace = True)

event_index = event_index.reset_index()


csv_data = pd.read_csv("../CSV_Data/_issue.csv")

csv_data['Event'] = event_index['Issue']

csv_data.to_csv("../Data/related_issue_event.csv")
df_issue = csv_data[csv_data['Issue'] == target_issue]




for i in event_list:
    person, organization, place = extractNER(df_issue[df_issue['Event'] == i])
    print("Event", event_title_list[i])
    print("- Person       :", person)
    print("- Organization :", organization)
    print("- Place        :", place)
    print("")






