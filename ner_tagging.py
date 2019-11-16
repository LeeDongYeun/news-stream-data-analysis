import numpy as np
import pandas as pd
import re
import os
import json

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
    text = text.replace(",", " ,")
    
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    
    return sentences

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

            while joined[i][1] != 0 and joined[i][1] != 1:
                ner.append(joined[i])
                i += 1
            
            word = " ".join([x[0] for x in ner])
            entity = ner[0][1]
#             entity = index_to_ner[ner[0][1]]
            ner_list.append((word, entity))
        else:
            i += 1
    
    return ner_list


def ner_extract_from_text(model, text, word_to_index):
    max_len = 70
    ner_dict = {'Geographical Entity':[], 'Organization':[], 'Person':[], 'Geopolitical Entity':[],
           'Time indicator':[], 'Artifact':[], 'Event':[], 'Natural Phenomenon':[]}
    
    for sentence in text:
        new_sentence = sentence.lower().split()

        new_X=[]
        for w in new_sentence:
            try:
              new_X.append(word_to_index.get(w,1))
            except KeyError:
              new_X.append(word_to_index['OOV'])
              # 모델이 모르는 단어에 대해서는 'OOV'의 인덱스인 1로 인코딩

        pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=max_len)

        p = model.predict(np.array([pad_new[0]]))
        p = np.argmax(p, axis=-1)

        ner_list = process_ner(new_sentence, p[0])

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
    
    return ner_dict


def main():
    nerDataPath = 'Data/entity-annotated-corpus/ner_dataset.csv'
    newsDataPath = 'Data/news/'
    model_weights = 'bi_lstm_crf_weight.h5'

    datapaths = os.listdir(newsDataPath)

    df = pd.DataFrame()

    for p in datapaths:
        with open(newsDataPath + p, 'r') as f:
            data = json.load(f)

        dataframe = pd.DataFrame.from_dict(data)
        df = df.append(dataframe)

    df = df.reset_index(drop=True)
    
    # word_to_index를 얻기 위한 작업
    nerData = load_data(nerDataPath)
    sentences, ner_tags = preprocess_data(nerData)

    src_tokenizer, _ = tozenizer(sentences, ner_tags)
    word_to_index = src_tokenizer.word_index

    # bi-LSTM CRF model
    model = biLSTM()
    print("loading weight to bi-LSTM model...")
    model.load_weights(model_weights)

    #
    ner_df = pd.dataFrame()
    for i in range(df.shape[0]):
        column = df.iloc[0]
        body = column[' body']
        sentence_split = split_into_sentences(body)

        ner_dict = ner_extract_from_text(model=model, text=sentence_split, word_to_index=word_to_index)

        val_list = list(ner_dict.values())
        joined_val_list = []
        for val in val_list:
            joined_val = ','.join(val)
            joined_val_list.append(joined_val)

        ner = pd.DataFrame(data=[joined_val_list], columns=list(ner_dict.keys()))
        ner_df.append(ner)

    ner_df = ner_df.reset_index(drop=True)

    merged_df = pd.merge(df, ner_df)
    merged_df.to_csv("Data/ner_result.csv", mode='w')


if __name__ == "__main__":
    main()


'''
text = ['The United States announced retaliatory sanctions on North Korea on Friday in response to the communist nation \'s alleged cyber-attacks on Sony Pictures , warning the actions are just the  " first aspect "  of its response .',
 'President Barack Obama signed an executive order authorizing additional sanctions on North Korean individuals and entities in response to the North \'s  " ongoing provocative , destabilizing , and repressive actions and policies , particularly its destructive and coercive cyber attack on Sony , "  the White House said in a statement .',
 "Three North Korean entities and 10 officials were named in the sanctions , including the Reconnaissance General Bureau , Pyongyang 's primary intelligence organization , accused of arms trading and other activities banned under U.N. resolutions , according to the Treasury Department .",
 'Though those sanctioned are barred from using the U.S. financial system and U.S. citizens are banned from doing business with them , the measures are considered largely symbolic because the North has already been under a string of international sanctions and those newly sanctioned are not believed to have any dealings with the U.S. " We take seriously North Korea \'s attack that aimed to create destructive financial effects on a U.S. company and to threaten artists and other individuals with the goal of restricting their\xa0right to free expression , "  the White House said "  .',
 "As the president has said , our response to North Korea 's attack against Sony Pictures Entertainment will be proportional , and will take place at a time and in a manner of our choosing .",
 'Today \'s actions are the first aspect of our response , "  it said .',
 'The FBI has determined that North Korea was behind the hack on Sony , confirming widespread suspicions pointing to the North that has expressed strong anger at a Sony movie ,  " The Interview , "  which involves a plot to assassinate North Korean leader Kim Jong-un .',
 'Obama has since vowed to  " respond proportionally "  to the attacks .',
 'North Korea has denied any responsibility , though it lauded the Sony hack as a  " righteous deed "  .',
 '" The order is not targeted at the people of North Korea , but rather is aimed at the government of North Korea and its activities that threaten the United States and others , "  Obama said in a letter to House of Representatives and Senate leaders .',
 'The two other newly sanctioned North Korean entities are Korea Mining Development and Trading Corp .',
 '(KOMID) and Korea Tangun Trading Corp .',
 'Eight of the 10 sanctioned individuals were KOMID officials stationed in Iran , Syria , Russia and Namibia .',
 "KOMID is the North 's primary arms dealer and main exporter of goods and equipment related to ballistic missiles and conventional weapons , according to the Treasury Department .",
 'The company was previously sanctioned by the U.S. and the United Nations , it said .',
 'Korea Tangun Trading Corp .',
 "is responsible for the procurement of commodities and technologies to support the North 's defense research and development program .",
 'The company was also a target of U.S. and U.N. sanctions , the department said .',
 'The sanctioned individuals include KOMID officials Kil Jong-hun , Kim Kwang-yon , Jang Song-chol , Kim Yong-chol , Jang Yong-son , Kim Kyu , Ryu Jin and Kang Ryong , as well as Yu Kwang-ho , a North Korean government official , and Kim Kwang-chun , a Tangun Trading Corp .',
 'official "  .',
 'Today \'s actions are driven by our commitment to hold North Korea accountable for its destructive and destabilizing conduct , "  Secretary of the Treasury Jacob Lew said in a statement .',
 '" Even as the FBI continues its investigation into the cyber-attack against Sony Pictures Entertainment , these steps underscore that we will employ a broad set of tools to defend U.S. businesses and citizens , and to respond to attempts to undermine our values or threaten the national security of the United States "  .',
 "The new sanctions also underline the confidence the U.S. has in blaming the North for the Sony hack despite growing doubts about the FBI 's finding among American cyber-security specialists .",
 "Last week , a cyber-security firm , Norse , was reported to have briefed the FBI on the result of its own investigation that it was not North Korea , but laid-off Sony staff members that disrupted Sony 's computer network .",
 'On Friday , Scott Borg , director and chief economist of the U.S. Cyber Consequences Unit , an independent , nonprofit research institute specializing on cyber-threats and risks , also said in a commentary on the CNBC website that the skills employed in the Sony hack were too sophisticated for the North .']

dataPath = 'Data/entity-annotated-corpus/ner_dataset.csv'
model_weights = 'bi_lstm_crf_weight.h5'

data = load_data(dataPath)
sentences, ner_tags = preprocess_data(data)

src_tokenizer, _ = tozenizer(sentences, ner_tags)
word_to_index = src_tokenizer.word_index

model = biLSTM()
print("loading weight to bi-LSTM model...")
model.load_weights(model_weights)


print(ner_extract_from_text(model, text, word_to_index))
'''


