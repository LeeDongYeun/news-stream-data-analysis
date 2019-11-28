import os
import json
import re
import string
import pandas as pd
import numpy as np
from collections import Counter

def lemma_NER(ner_list):
    except_set = ['(yonhap)', 'yonhap', '']
    ner_list = [x for x in ner_list if x not in except_set]

    punc = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    punc += "“”’"
    ner_list = [word.translate(word.maketrans(punc, " "*len(punc))).strip() for word in ner_list]
    
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
    
