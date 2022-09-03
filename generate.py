from keras.models import load_model
import numpy as np
from train import Train


class Generate(Train):
    def __init__(self, model_path, max_words, number_of_input_words):
        super().__init__(max_words=max_words, number_of_input_words=number_of_input_words)
        self.model = load_model(model_path)
        self.previous_text = ''

    def get_command(self):
        self.command = input().split()
        if '--input-dir' in self.command:
            self.input_dir = self.command[self.command.index('--input-dir') + 1]
        if '--model' in self.command:
            self.model_path = self.command[self.command.index('--model') + 1]
        if '--prefix' in self.command:
            self.previous_text = self.command[self.command.index('--prefix') + 1]
        print(self.input_dir, self.model_path)

    def generate_sequence(self, seq_len=20):
        if not self.previous_text:
            previous_text = np.random.choice()
        self.tokenizer.fit_on_texts([self.text])
        data = self.tokenizer.texts_to_sequences([previous_text])[0]
        print(data)
        res = ""
        for i in range(seq_len):
            x = data[i:i + self.number_of_input_words]
            input_sequence = np.expand_dims(x, axis=0)
            print(len(input_sequence))
            pred = self.model.predict(input_sequence, verbose=0)
            index = pred.argmax(axis=1)[0]
            data.append(index)
            res += " " + self.tokenizer.index_word[index]
        return previous_text + res


def main():
    generate_ = Generate(model_path="model.h5", max_words=1500, number_of_input_words=3)
    generate_.get_command()
    print(generate_.generate_sequence('привет как ты а'))


if __name__ == "__main__":
    main()

