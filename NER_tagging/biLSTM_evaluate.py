import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from biLSTM import load_data, preprocess_data, tozenizer, biLSTM


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


def sequences_to_tag(sequences): # 예측값을 index_to_tag를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences: # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence: # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred) # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O")) # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


def main():
    baseDataPath = 'Data/entity-annotated-corpus/ner_dataset.csv'
    evaluationeDataPath = ''
    model_weights = 'bi_lstm_crf_weight.h5'

    baseData = load_data(baseDataPath)
    base_sentences, base_ner_tags = preprocess_data(baseData)

    src_tokenizer, tar_tokenizer = tozenizer(base_sentences, base_ner_tags)
    word_to_index = src_tokenizer.word_index
    index_to_ner = tar_tokenizer.index_word
    index_to_ner[0]='PAD'

    data = load_data(evaluationeDataPath)
    sentences, ner_tags = preprocess_data(data)




    # bi-LSTM CRF model
    model = biLSTM()
    print("loading weight to bi-LSTM model...")
    model.load_weights(model_weights)

    print("\n test accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))


    return 0

if __name__ == '__main__':
    main()