import os
import json
import re
import string
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import ast


def text_process(text):
    #number removal
    if text == -2:
        return ''
        
    body = re.sub(r'\d+', '', text)
    
    #punctuation removal i.e. [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
#     punc = string.punctuation
#     punct_mapping = {"_":" ", "'":" "}
#     punc += "“”’"
    punc = "/-'?!,#$%\'()*+-/:;<=>@\\^_`{|}~[]" + '""“”’'
 
#     punc = re.sub("-","", punc)
    body = body.translate(body.maketrans(punc, " " * len(punc)))
    
    #text lower
    body = body.lower()
    
    #multi-word tokenize
    multi_word_list = [('north', 'korea'), ('south', 'korea'), ('north', 'korean'), ('south', 'korean'),
                      ('kim', 'jong', 'un'), ('park', 'geun', 'hye')]
    tokenizer = MWETokenizer()
    for mw in multi_word_list:
        tokenizer.add_mwe(mw)
    text = tokenizer.tokenize(body.split())
    
    #stopwort removal
    stopset = set(stopwords.words('english'))
#     text = word_tokenize(body)
    text = [x for x in text if x not in stopset]
    text = [word for word in text if len(word) > 3]
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_text = [lemmatizer.lemmatize(x) for x in text]
    
    return lemma_text


def make_text_topic_list(df, issue_num):
    text_topic_list = []
    for j in range(issue_num):
        temp_list = [df['clean_title_list'].iloc[i] for i in range(len(df)) if df.loc[i,'Issue'] == j]
        text_topic_list.append(temp_list)
    
    # Find most overllaped sequence
    MAX_SEQUENCE_LENGTH = 8
    total_dict_list = []
    
    for _ in range(issue_num):
        dict_list = []
        for _ in range(MAX_SEQUENCE_LENGTH + 1):
            dict_list.append(dict())
        total_dict_list.append(dict_list)
    for k in range(issue_num):
        topic_list = text_topic_list[k]
        dict_list = total_dict_list[k]
        for sequence_length in range(1, MAX_SEQUENCE_LENGTH + 1):

            for single_topic_list in topic_list:

                for i in range( len(single_topic_list) - sequence_length):

                    if str( single_topic_list[i:i+sequence_length] ) not in dict_list[sequence_length]:
                        dict_list[sequence_length][str( single_topic_list[i:i+sequence_length] )] = 0
                    dict_list[sequence_length][str( single_topic_list[i:i+sequence_length] )] += 1
    
    return total_dict_list


def tf_idf_result(corpus):
    result_set_list = []
    vectorizer1 = CountVectorizer()
    vectorizer2 = TfidfVectorizer()
    
    tf = vectorizer1.fit_transform(corpus)
    tf_idf = vectorizer2.fit_transform(corpus)

    tf = tf.toarray()
    tf_idf = tf_idf.toarray()

    features = vectorizer1.get_feature_names()

    tf_dataFrame = pd.DataFrame(tf, columns=features)
    tf_idf_dataFrame = pd.DataFrame(tf_idf, columns=features)

    tf_dataFrame = tf_dataFrame.sum()
    tf_dataFrame = pd.DataFrame(tf_dataFrame)
    tf_dataFrame = tf_dataFrame.sort_values([0], ascending=[False])

    tf_idf_dataFrame = tf_idf_dataFrame.sum()
    tf_idf_dataFrame = pd.DataFrame(tf_idf_dataFrame)
    tf_idf_dataFrame = tf_idf_dataFrame.sort_values([0], ascending=[False])
    
    tf_idf_dataFrame = tf_idf_dataFrame.reset_index()
    
    for i in range(tf_idf_dataFrame.shape[0]):
        x = tf_idf_dataFrame.iloc[i]['index']
        y = tf_idf_dataFrame.iloc[i][0]
        result_set_list.append((x,float(y)))
    
    return result_set_list


def issue_name(df, issue, candidate_topic_list, min_sentence_length):
    df_issue = df[df['Issue'] == issue]
    
    corpus = df_issue['clean_title']
    _list = tf_idf_result(corpus)
    
    test_list = []
    max_title = [([], 0)]
    for j in range(min_sentence_length, 5):
            items = candidate_topic_list[issue][j].items()
            for key, value in items:
                key = ast.literal_eval(key)

                if value >=3 :
                    tf_idf = 0
                    for num in range(len(key)):
                        if key[num] in ['u.s.', 'u.k.', 'n.k.', 'u.n.','‘comfort','korea‘s','p.m.','govt.']:
                            tf_idf += 0
                        else:
                            tf_idf += dict(_list)[key[num]]
                    tf_idf /= len(key)
                    if tf_idf > max_title[0][1]:
                        max_title[0] = (key, tf_idf)
    
    return max_title[0][0]


number_of_topic = 125
path = '../Data/2-2 result/related_issue_event.csv'
df = pd.read_csv(path)

df = df.drop('Unnamed: 0', axis=1)
df = df.drop('Unnamed: 0.1', axis=1)
df = df.fillna(-2)

df['clean_title'] = pd.DataFrame(df['title'].apply(lambda x: ' '.join(text_process(x))))
df['clean_title_list'] = pd.DataFrame(df['title'].apply(lambda x: text_process(x)))

top_issues = df['Issue'].value_counts().keys()[:10]

candidate_topic_list = make_text_topic_list(df, number_of_topic)

for issue in top_issues:
    issue = int(issue)
    issueName = issue_name(df, issue, candidate_topic_list, 2)
    print(issue, issueName)