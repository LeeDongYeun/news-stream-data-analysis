import numbers as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from biLSTM import load_data, preprocess_data, tozenizer, biLSTM


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
    dataPath = 'Data/entity-annotated-corpus/ner_dataset.csv'
    model_weights = 'bi_lstm_crf_weight.h5'

    data = load_data(dataPath)
    sentences, ner_tags = preprocess_data(data)

    src_tokenizer, tar_tokenizer = tozenizer(sentences, ner_tags)

    vocab_size = len(src_tokenizer.word_index) + 1
    tag_size = len(tar_tokenizer.word_index) + 1

    X_train = src_tokenizer.texts_to_sequences(sentences)
    y_train = tar_tokenizer.texts_to_sequences(ner_tags)

    index_to_word = src_tokenizer.index_word
    index_to_ner = tar_tokenizer.index_word
    index_to_ner[0]='PAD'

    max_len = 70
    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

    y_train = to_categorical(y_train, num_classes=tag_size)
    y_test = to_categorical(y_test, num_classes=tag_size)

    print('훈련 샘플 문장의 크기 : {}'.format(X_train.shape))
    print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
    print('테스트 샘플 문장의 크기 : {}'.format(X_test.shape))
    print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))

    # bi-LSTM CRF model
    model = biLSTM()
    print("loading weight to bi-LSTM model...")
    model.load_weights(model_weights)

    print("\n test accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))


    return 0

if __name__ == '__main__':
    main()