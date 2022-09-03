from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras.layers import Dense, SimpleRNN, Embedding
from keras.models import Sequential
import numpy as np


def get_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def get_data(train_text, number_of_input_words=3, max_words=1500):
    tokenizer = Tokenizer(num_words=max_words, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                          lower=True, split=' ', char_level=False)
    tokenizer.fit_on_texts([train_text])

    data = tokenizer.texts_to_sequences([train_text])
    res = np.array(data[0])

    print(res.shape, res)

    x_train = np.array([res[i:i + number_of_input_words] for i in range(res.shape[0] - number_of_input_words)])
    y_train = to_categorical(res[number_of_input_words:], num_classes=max_words)

    # print(x_train, y_train)

    return tokenizer, x_train, y_train


def build_model(number_of_input_words=3, max_words=1500):
    model = Sequential()
    model.add(Embedding(max_words, 256, input_length=number_of_input_words))
    model.add(SimpleRNN(64))
    model.add(Dense(max_words, activation='softmax'))
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='adam')
    return model


def fit_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=16, epochs=20)
    return model


def get_words(model, tokenizer_, previous_text, str_len=20, number_of_input_words=3):
    data = tokenizer_.texts_to_sequences([previous_text])[0]
    res = ""
    for i in range(str_len):
        x = data[i:i + number_of_input_words]
        input_sequence = np.expand_dims(x, axis=0)
        print(len(input_sequence))
        pred = model.predict(input_sequence, verbose=0)
        index = pred.argmax(axis=1)[0]
        data.append(index)
        res += " " + tokenizer_.index_word[index]
    return previous_text + res


text = get_text("documents/harry_potter.txt")
tokenizer, x_train, y_train = get_data(text, 3)
model = build_model()
model = fit_model(model, x_train, y_train)
print(get_words(model, tokenizer, "Могу я попрощаться с"))

