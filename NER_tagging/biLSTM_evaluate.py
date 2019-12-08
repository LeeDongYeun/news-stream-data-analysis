import numpy as np
import pandas as pd
import os
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from biLSTM import load_data, preprocess_data, tozenizer, biLSTM


def texts_to_sequences(sentences, word_to_index):
    seq_list = []
    for sentence in sentences:
        new_X = []
        for w in sentence:
            try:
              new_X.append(word_to_index.get(w,1))
            except KeyError:
              new_X.append(word_to_index['OOV'])
        
        seq_list.append(new_X)
    
    return seq_list


def sequences_to_tag(sequences, index_to_ner):
    result = []
    for sequence in sequences:
        temp = []
        for pred in sequence:
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(temp)
    return result


def main():
    baseDataPath = '../Data/entity-annotated-corpus/ner_dataset.csv'
    evaluationeDataPath = '../Data/evaluation_data/merge.csv'
    model_weights = 'bi_lstm_crf_weight.h5'

    baseData = load_data(baseDataPath)
    base_sentences, base_ner_tags = preprocess_data(baseData)

    src_tokenizer, tar_tokenizer = tozenizer(base_sentences, base_ner_tags)
    word_to_index = src_tokenizer.word_index
    tag_size = len(tar_tokenizer.word_index) + 1
    index_to_ner = tar_tokenizer.index_word
    index_to_ner[0]='PAD'
    
    '''
    datapaths = os.listdir(evaluationeDataPaths)
    data = pd.DataFrame()
    for p in datapaths:
        _data = pd.read_csv(evaluationeDataPaths + p)
        data.append(_data)
        '''
    data = pd.read_csv(evaluationeDataPath)
    sentences, ner_tags = preprocess_data(data)

    X = texts_to_sequences(sentences, word_to_index)
    y = tar_tokenizer.texts_to_sequences(ner_tags)

    max_len = 70
    X = pad_sequences(X, padding='post', maxlen=max_len)
    y = pad_sequences(y, padding='post', maxlen=max_len)
    y = to_categorical(y, num_classes=tag_size)

    # bi-LSTM CRF model
    model = biLSTM()
    print("loading weight to bi-LSTM model...")
    model.load_weights(model_weights)

    y_predicted = model.predict(X)
    pred_tags = sequences_to_tag(y_predicted, index_to_ner)
    test_tags = sequences_to_tag(y, index_to_ner)

    print(classification_report(test_tags, pred_tags))
    print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

    return 0

if __name__ == '__main__':
    main()