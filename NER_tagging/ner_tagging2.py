import os
import json
import re
import string
import pandas as pd
import numpy as np
import ast

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from biLSTM import load_data, preprocess_data, tozenizer, biLSTM


def split_into_sentences(text):
    """
    This function can split the entire text of Huckleberry Finn into sentences in about 0.1 seconds
    and handles many of the more painful edge cases that make sentence parsing non-trivial 
    e.g. "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining 
    Nike Inc. as an engineer. He also worked at craigslist.org as a business analyst."
    """
    
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace("."," .<stop>")#     text = text.replace(".",".<stop>")
    text = text.replace("?"," ?<stop>")#     text = text.replace("?","?<stop>")
    text = text.replace("!"," !<stop>")#     text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    
    text = text.replace('"', ' " ')
    text = text.replace("\'s", " \'s")
    text = text.replace("‘s", " ‘s")
    text = text.replace("’s", " ’s")
    text = text.replace(",", " ,")
    
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    
    return sentences


def f(x):
    return len(x['clean_body']) * [x['index']]


def make_sentence_list(text):
    sentence_list = []
    for sentence in text:
        new_sentence = sentence.lower().split()
        sentence_list.append(new_sentence)
    return sentence_list


def make_index_list(text, word_to_index):
    max_len = 70
    indexed_list = []
    for sentence in text:
        new_sentence = sentence.lower().split()

        new_X=[]
        for w in new_sentence:
            try:
              new_X.append(word_to_index.get(w,1))
            except KeyError:
              new_X.append(word_to_index['OOV'])

        pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=max_len)
        indexed_list.append(pad_new[0])
    return indexed_list


def ner_tag(dataframe, word_to_index, model, filename):
    df = dataframe.copy()
    df['clean_body'] = pd.DataFrame(df[' body'].apply(lambda x: split_into_sentences(x)))

    df['word_index_list'] = pd.DataFrame(df['clean_body'].apply(lambda x: make_index_list(x, word_to_index)))
    df['sentence_list'] = pd.DataFrame(df['clean_body'].apply(lambda x: make_sentence_list(x)))
    df['doc_index_list'] = df.apply(f, axis=1)

    word_index_list = []
    sentence_list = []
    doc_index_list = []

    for w in df['word_index_list']:
        word_index_list += w

    for s in df['sentence_list']:
        sentence_list += s

    for d in df['doc_index_list']:
        doc_index_list += d

    print("predicting", filename)
    p = model.predict(np.array(word_index_list))
    p = np.argmax(p, axis=-1)

    ner_df = pd.DataFrame(p)
    ner_df['index'] = doc_index_list
    ner_df['sentence'] = sentence_list

    ner_df.to_csv(filename)

    return ner_df


def process_ner(sentence, bio):
    
    joined = []
    for w, pred in zip(sentence, bio):
        joined.append((w,pred))
    
    i = 0
    ner_list = []
    while i < len(joined):
        if joined[i][1] != 0 and joined[i][1] != 1:
            ner = []
            ner.append(joined[i])
            i += 1
            
            if i < len(joined):
                while joined[i][1] != 0 and joined[i][1] != 1:
                    ner.append(joined[i])
                    i += 1
                    if i >= len(joined):
                        break
            
            word = " ".join([x[0] for x in ner])
            entity = ner[0][1]
#             entity = index_to_ner[ner[0][1]]
            ner_list.append((word, entity))
        else:
            i += 1
    
    return ner_list


def word_index_to_NER_df(ner_df):
    ner_df2 = pd.DataFrame()

    for i in range(ner_df.shape[0]):
        # if i % 500 == 0:
            # print(i)
        
        
        ner_dict = {'Geographical Entity':[], 'Organization':[], 'Person':[], 'Geopolitical Entity':[],
            'Time indicator':[], 'Artifact':[], 'Event':[], 'Natural Phenomenon':[]}
        
        row = ner_df.iloc[i]
        
        bio = list(row.iloc[1:71])
        sentence = row['sentence']
        sentence = ast.literal_eval(sentence)
        
        ner_list = process_ner(sentence, bio)
        
        for entity in ner_list:
            if entity[1] == 2:
                ner_dict['Geographical Entity'].append(entity[0])
            elif entity[1] == 3:
                ner_dict['Time indicator'].append(entity[0])
            elif entity[1] == 4:
                ner_dict['Organization'].append(entity[0])
            elif entity[1] == 6:
                ner_dict['Person'].append(entity[0])
            elif entity[1] == 8:
                ner_dict['Geopolitical Entity'].append(entity[0])
            elif entity[1] == 11:
                ner_dict['Artifact'].append(entity[0])
            elif entity[1] == 12:
                ner_dict['Event'].append(entity[0])
            elif entity[1] == 15:
                ner_dict['Natural Phenomenon'].append(entity[0])
        
        val_list = list(ner_dict.values())
        joined_val_list = []
        for val in val_list:
            joined_val = ','.join(val)
            joined_val_list.append(joined_val)

        ner = pd.DataFrame(data=[joined_val_list], columns=list(ner_dict.keys()))
        ner_df2 = ner_df2.append(ner)
        
    ner_df2 = ner_df2.reset_index(drop=True)
    ner_df2['index'] = ner_df['index']

    cols = ner_df2.iloc[:, 0:-1].columns

    data = []
    for i in range(ner_df.iloc[-1]['index'] + 1):
        index_df = ner_df2[ner_df2['index'] == i]
        l = []
        for col in cols:
            x = index_df[col]
            x = [k for k in x if k != '']
            x = ','.join(x)

            l.append(x)
        data.append(l)

    result = pd.DataFrame(data=data, columns=list(cols))
    result['index'] = result.index

    return result


def main():
    nerDataPath = '../Data/entity-annotated-corpus/ner_dataset.csv'
    newsDataPath = '../Data/news/'
    resultDataPath = '../Data/ner_tagged_news/'
    model_weights = 'bi_lstm_crf_weight.h5'

    # word_to_index를 얻기 위한 작업
    nerData = load_data(nerDataPath)
    sentences, ner_tags = preprocess_data(nerData)

    src_tokenizer, tar_tokenizer = tozenizer(sentences, ner_tags)
    word_to_index = src_tokenizer.word_index
    index_to_ner = tar_tokenizer.index_word
    index_to_ner[0]='PAD'

    model = biLSTM()
    print("Loading weight to bi-LSTM model...")
    model.load_weights(model_weights)

    datapaths = os.listdir(newsDataPath)
    for p in datapaths:
        tempFileName = 'temp_' + p.split('.')[0] + '.csv'
        resultFileName = 'NER_' + p.split('.')[0] + '.csv'
        with open(newsDataPath + p, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data)
        df = df.reset_index(drop=True)
        df['index'] = df.index
        
        ner_df = ner_tag(df, word_to_index, model, tempFileName)
        print(tempFileName, "file saved.")
        ner_df = pd.read_csv(tempFileName)

        result = word_index_to_NER_df(ner_df)

        df = pd.merge(df, result, how='right')
        df.to_csv(resultDataPath + resultFileName)
        print(resultFileName, "file saved.")
    
    datapaths = os.listdir(resultDataPath)
    concat = pd.DataFrame()
    for p in datapaths:
        ner_df = pd.read_csv(datapaths + p)
        concat = concat.append(ner_df)
    concat.to_csv('../Data/ner_tagged.csv')
    
    print("Done.")


if __name__ == '__main__':
    main()