import shopping_data

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten
from keras.layers import LSTM
import chinese_vec
import numpy as np


def main():
    x_train, y_train, x_test, y_test = shopping_data.load_data()
    vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)

    x_train_index = shopping_data.word2Index(x_train, word_index)
    x_test_index = shopping_data.word2Index(x_test, word_index)

    maxlen = 25
    x_train_index = sequence.pad_sequences(x_train_index, maxlen = maxlen)
    x_test_index = sequence.pad_sequences(x_test_index, maxlen = maxlen)

    # embedded matrix
    word_vecs = chinese_vec.load_word_vecs()
    embedding_matrix = np.zeros((vocalen,300))
    for word, i in word_index.items():
        embedding_vector = word_vecs.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    model = Sequential()
    model.add(Embedding(trainable = False,weights=[embedding_matrix], input_dim = vocalen, output_dim = 300, input_length = maxlen))
    model.add(LSTM(128), return_sequences=True)
    model.add(LSTM(128))

    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train_index, y_train,
              batch_size=512,
              epochs=200)


    score, acc = model.evaluate(x_test_index, y_test)

    print('Test score: ', score)
    print('Test accuracy: ', acc)




if __name__ == '__main__':
    main()

