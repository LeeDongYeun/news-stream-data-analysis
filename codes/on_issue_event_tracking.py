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
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast 
from collections import Counter


pd.set_option('display.max_colwidth', -1)
np.random.seed(222)

Issuenum = 78

def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)
    

def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def concat(*str):
    return " ".join(str)

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

def make_event_list(df, similarity):
    event_list = []
    visit = [0 for _ in range(len(df))]
    for i in range(len(df)-1):
        add = False
        if visit[i] == 0:    
            start_time = df.iloc[i][' time']
            until_7day_list = []
            until_7day_list.append((i, " ".join(df.iloc[i][' body'])))
            for j in range(i+1, len(df)):
                end_time = df.iloc[j][' time']

                if (end_time - start_time).days <= 7:
                    until_7day_list.append((j," ".join(df.iloc[j][' body'])))
                else:
                    break

            on_event_issue_set = set()
            on_event_issue_set.add(i)
            for k in range(1,len(until_7day_list)):
                if visit[until_7day_list[k][0]] == 1:
                    continue    

                for index in on_event_issue_set:
                    all_above = True
                    if get_cosine_sim(" ".join(df.iloc[index][' body']), until_7day_list[k][1])[0][1] < similarity:
                        all_above = False
                        break

                if all_above:    
                    on_event_issue_set.add(until_7day_list[k][0])
                    visit[until_7day_list[k][0]] = 1
                    add = True

        if add:
            visit[i] = 1
            event_list.append(on_event_issue_set)
    return event_list


def make_onissue_event_list(df, event_list, similarity):
    set_list = []
    set_visit = [0 for _ in range(len(event_list))]
    for i in range(len(event_list)-1):
        if set_visit[i]==1:
            continue
        target_set = event_list[i]
        target_sentence = ""
        for index in target_set:
            target_sentence =concat(target_sentence, str(df.iloc[index][' body']))

        target_sentence = target_sentence[1:]

        concat_list = []
        concat_index_list = []
        temp_target_set = target_set
        for j in range(i+1, len(event_list)):
            if set_visit[j]==1:
                continue
            compare_set = event_list[j]
            compare_sentence = ""
            for index in compare_set:
                compare_sentence = concat(compare_sentence, str(df.iloc[index][' body']))
            compare_sentence = compare_sentence[1:]

            if get_cosine_sim(target_sentence, compare_sentence)[0][1] >= similarity:

                if max(df.iloc[list(target_set)][' time']) >= min(df.iloc[list(compare_set)][' time']):
                    target_sentence = concat(target_sentence, " ".join(compare_sentence))
                    temp_target_set = temp_target_set.union(compare_set)
                    if not concat_index_list:
                        concat_index_list.append(i)
                    concat_index_list.append(j)
                else:
                    if not temp_target_set in concat_list:
                        concat_list.append(temp_target_set)
                    concat_list.append(compare_set)
                    concat_index_list.append(j)
                    for index in concat_index_list:                    
                        set_visit[index] = 1
        if concat_list:
            set_list.append(concat_list)
    return set_list


def choose_representative(df, temp_frame, index_list):
    
    max_score = 0
    max_score_sentence_index = -1
    body_list = []
    
    for index in index_list:
        body_list.append((index, " ".join(temp_frame.iloc[index][' body'])))
    for index_n_body in body_list:
        index, body = index_n_body
        score = 0
        for index_n_comp_body in body_list:
            comp_index, comp_body = index_n_comp_body
            if index!=comp_index:
                cosine_sim = get_cosine_sim(body, comp_body)[0][1]
                score += cosine_sim
        score /= len(body_list) - 1
        if score > max_score:
            max_score = score
            max_score_sentence_index = index
    return max_score_sentence_index, df[df["Unnamed: 0"]==temp_frame.iloc[max_score_sentence_index]['Doc_num']]['title'], temp_frame.iloc[max_score_sentence_index][' time']


text = pd.read_csv("../CSV_Data/preprocessed_data.csv")
text[' body'] = text[' body'].apply(ast.literal_eval)

temp_frame = pd.DataFrame()
for i in range(len(text)):
    if text.iloc[i]["Issue"]==Issuenum:
        temp_frame = temp_frame.append(text.iloc[i], ignore_index=True)
        
text[' time'] = pd.to_datetime(text[' time'])
text = text.sort_values(by=' time', ascending=True)

temp_frame[' time'] = pd.to_datetime(temp_frame[' time']).dt.date
temp_frame = temp_frame.sort_values([' time'])

# Make event list
event_list = make_event_list(temp_frame, 0.5)

# Make on-issue event list
onissue_event_list = make_onissue_event_list(temp_frame, event_list, 0.7)

# Print some titles

df = pd.read_csv("../CSV_Data/_issue.csv")
df["Event"] = ""
df.head()

for j, elem in enumerate(onissue_event_list):
    if len(elem)>=5:
        if j!=0:
            print("********************************************************************************")
        for i, ele in enumerate(elem):
            ele = sorted(ele)
            index, title, time = choose_representative(df, temp_frame, ele)
            print(title.to_string(index=False))
            print(time)
            pop_frame = pd.DataFrame()
            for e in ele:
                real_index = int(temp_frame.iloc[e]['Doc_num'])
                pop_frame = pop_frame.append(df.iloc[real_index])
            person, organization, place = extractNER(pop_frame)
            print("- Person       :", person)
            print("- Organization :", organization)
            print("- Place        :", place)

#             for e in ele:
                # print(temp_frame.iloc[e]['title'] , " ,", temp_frame.iloc[e][' time'])
                
            if i!=len(elem)-1:
                print()
                print(" |")
                print(" V")
                print()

# Export with tagged event number
for j, elem in enumerate(onissue_event_list):
    for i, ele in enumerate(elem):
        for index in ele:
            df.iloc[int(temp_frame.iloc[index]['Doc_num']), df.columns.get_loc("Event")] = j

df.to_csv("../CSV_Data/onissue_event.csv")