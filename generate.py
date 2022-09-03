from keras.models import load_model
import numpy as np
from train import Train


class Generate(Train):
    def __init__(self, model_path):
        super().__init__(max_words=1500, number_of_input_words=5)
        self.model = load_model(model_path)

    def generate_sequence(self, length):
        input_sequence = input()
        data = self.tokenizer.texts_to_sequences([input_sequence])[0]
        res = ""
        for i in range(length):
            x = data[i:i + self.number_of_input_words]
            input_sequence = np.expand_dims(x, axis=0)
            if len(input_sequence[0]) < self.number_of_input_words:
                input_sequence[0] = input_sequence[0].tolist() + [0 for _ in range(5 - len(input_sequence))]
            pred = self.model.predict(input_sequence)
            index = pred.argmax(axis=1)[0]
            data.append(index)
            print(index)
            res += " " + self.tokenizer.index_word[index]
        return input_sequence + res
