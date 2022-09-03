from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Dense, SimpleRNN, Embedding
from keras.models import Sequential
import numpy as np
import os


class Train:
    def __init__(self, max_words, number_of_input_words):
        self.max_words = max_words
        self.number_of_input_words = number_of_input_words
        self.command = ''
        self.input_dir = ''
        self.model_path = ''
        self.text = ''
        self.number_of_input_words = 1
        self.tokenizer = Tokenizer(num_words=max_words, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                              lower=True, split=' ', char_level=False)

    def get_command(self):
        self.command = input().split()
        if '--input-dir' in self.command:
            self.input_dir = self.command[self.command.index('--input-dir') + 1]
        if '--model' in self.command:
            self.model_path = self.command[self.command.index('--model') + 1]

    def get_texts(self):
        filenames = os.listdir(self.input_dir)
        for filename in filenames:
            path = os.path.join(self.input_dir, filename)
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
            self.text += text
        print(self.text)

    def get_data(self):
        self.tokenizer.fit_on_texts([self.text])

        data = self.tokenizer.texts_to_sequences([self.text])
        res = np.array(data[0])

        x_train = np.array([res[i:i + self.number_of_input_words] for i in range(res.shape[0] - self.number_of_input_words)])
        y_train = to_categorical(res[self.number_of_input_words:], num_classes=self.max_words)

        return x_train, y_train

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_words, 256, input_length=self.number_of_input_words))
        model.add(SimpleRNN(128, return_sequences=True))
        model.add(SimpleRNN(64))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.max_words, activation='softmax'))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='adam')
        return model

    def fit_model(self, model, x_train, y_train, epochs=20):
        model.fit(x_train, y_train, batch_size=16, epochs=epochs)
        model.save(f'model/{self.model_path}')
        return model


def main():
    train_ = Train(1500, 3)
    train_.get_command()
    train_.get_texts()
    x_train, y_train = train_.get_data()
    model = train_.build_model()
    train_.fit_model(model, x_train, y_train)


if __name__ == "__main__":
    main()
