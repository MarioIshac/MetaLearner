import os

import numpy as np
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical

import tokenize

def get_python_filepaths(directory):
    for path, directories, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                joined_file_path = os.path.join(path, file)
                yield joined_file_path


class Token:
    def __init__(self, tokenString, tokenType):
        self.tokenString = tokenString
        self.tokenType = tokenType

    def __eq__(self, other):
        tokenStringsEqual = self.tokenString == other.tokenString
        tokenTypesEqual = self.tokenType == other.tokenType

        return tokenStringsEqual and tokenTypesEqual

    def __hash__(self):
        return hash((self.tokenString, self.tokenType))

    def __repr__(self):
        return "Token(tokenString=%s, tokenType=%s)" % (self.tokenString, self.tokenType)

empty_token = Token(None, -3)
start_of_file_token = Token(None, -2)
end_of_file_token = Token(None, -1)

# Represents number of tokens that we consider at a time within the recurrent neural network. The back propagation will
# go this many words far back in the sequence
step_input_size = 5

# Number of tokens to skip between iterations.
step_skip_size = 1

# Numbers of iterations to perform before performing parameter updates in model.
mini_batch_size = 1

def build_token_vocabulary(tokens):
    unique_tokens = set(tokens)
    unique_token_indices = range(len(unique_tokens))

    return unique_tokens, unique_token_indices


def get_token_to_index(unique_tokens, unique_token_indices):
    token_to_index = dict(zip(unique_tokens, unique_token_indices))

    return token_to_index


def get_index_to_token(unique_tokens, unique_token_indices):
    index_to_token = dict(zip(unique_token_indices, unique_tokens))

    return index_to_token


def read_tokens(filepaths):
    tokens = []

    for filename in filepaths:
        with open(filename, "r") as file:
            for _ in range(step_input_size - 1):
                tokens.append(empty_token)

            tokens.append(start_of_file_token)

            for token in tokenize.generate_tokens(file.readline):
                token = Token(token.string, token.type)

                if token.tokenType == tokenize.COMMENT:
                    continue

                tokens.append(token)

            tokens.append(end_of_file_token)

    return np.array(tokens)

def load_data():
    # get the data paths
    train_path = "data/keras-master"
    test_path = "data/keras-master"

    train_file_paths = get_python_filepaths(train_path)
    test_file_paths = get_python_filepaths(test_path)

    train_tokens = read_tokens(["meta_learner.py"])
    test_tokens = read_tokens(["meta_learner.py"])

    unique_tokens, unique_token_indices = build_token_vocabulary(train_tokens)
    token_to_index = get_token_to_index(unique_tokens, unique_token_indices)
    index_to_token = get_index_to_token(unique_tokens, unique_token_indices)

    train_data = [token_to_index[token] for token in train_tokens if token in unique_tokens]
    test_data = [token_to_index[token] for token in test_tokens if token in unique_tokens]

    token_vocabulary = len(unique_tokens)

    return train_data, test_data, token_vocabulary, token_to_index, index_to_token


train_data, test_data, token_vocabulary_size, token_to_index, index_to_token = load_data()

empty_token_index = token_to_index[empty_token]
start_token_index = token_to_index[start_of_file_token]
end_of_file_token_index = token_to_index[end_of_file_token]

print(train_data)

class KerasBatchGenerator():
    def __init__(self, data, num_steps, mini_batch_size, token_vocabulary, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size
        self.vocabulary = token_vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.mini_batch_size, self.num_steps))
        print("MB " + str(self.mini_batch_size))
        print("NS " + str(self.num_steps))
        print("VOCAB " + str(self.vocabulary))

        y = np.zeros((self.mini_batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.mini_batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

train_data_generator = KerasBatchGenerator(train_data, step_input_size, mini_batch_size, token_vocabulary_size,
                                           step_skip_size)

hidden_size = 500
use_dropout = True
model = Sequential()
model.add(Embedding(token_vocabulary_size, hidden_size, input_length=step_input_size))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(token_vocabulary_size)))
model.add(Activation('softmax'))

from keras.optimizers import Adam

model_optimizer = Adam(lr=0.001, decay=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['categorical_accuracy'])

num_epochs = 200

# Number of mini batches that will be yielded per epoch. This is required to know how many times we call
# KerasBatchGenerator#generate() per epoch
mini_batches_per_epoch = len(train_data) // (step_skip_size * mini_batch_size)


def train():
    model.fit_generator(train_data_generator.generate(), mini_batches_per_epoch, num_epochs)

from sys import maxsize

def sample(fileName, maxLength=maxsize):
    token_indices = [empty_token_index for _ in range(step_input_size - 1)]
    token_indices.append(start_token_index)

    while token_indices[-1] != end_of_file_token_index and len(token_indices) < maxLength:
        previous_token_indices = token_indices[-step_input_size:]

        # model predict includes a dimension indexed by mini batch index regardless of mini batchs size
        # (in this case 1)
        next_tokens_prediction_vector_per_batch = model.predict_on_batch(np.array([previous_token_indices]))

        # so get first element to get desired prediction vector
        next_tokens_prediction_vector = next_tokens_prediction_vector_per_batch[0]

        # then get the last predicted token since all but the last token have already been appended
        # to the sample tokens
        next_token_prediction_vector = next_tokens_prediction_vector[-1]

        next_token_indices = range(token_vocabulary_size)

        next_token_index = np.random.choice(next_token_indices, p=next_token_prediction_vector)

        token_indices.append(next_token_index)

    tokens = [index_to_token[token_index] for token_index in token_indices]
    writable_tokens = [token for token in tokens if token.tokenString != None]

    print(writable_tokens)

    with open(fileName, "w") as file:
        for writable_token_index in range(len(writable_tokens)):
            current_writable_token = writable_tokens[writable_token_index]
            next_writable_token_index = writable_token_index + 1

            # Indicates we are done writing to file
            if next_writable_token_index == len(writable_tokens):
                return

            next_writable_token = writable_tokens[writable_token_index + 1]

            file.write(current_writable_token.tokenString)

            # If there are two consecutive name tokens, separate them with space
            if current_writable_token.tokenType == tokenize.NAME and next_writable_token.tokenType == tokenize.NAME:
                file.write(" ")

def test():
    # We specify mini batch size here to be 1 because we want to be able to generate 1 character at a time as opposed
    # to mini_batch_size characters at a time. This allows the user to specify number of character to generate and test
    # rather than number of mini batches.
    example_training_generator = KerasBatchGenerator(train_data, step_input_size, 1, token_vocabulary_size,
                                                     skip_step=1)

    num_predict = 10
    true_print_out = []
    pred_print_out = []
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        print(type(data[0]))
        print(data[0].shape)
        print(data[0])
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, step_input_size - 1, :])
        true_print_out.append(index_to_token[train_data[step_input_size + i]])
        pred_print_out.append(index_to_token[predict_word])
    print("Actual")
    print([token.tokenString for token in true_print_out])
    print("Predicted")
    print([token.tokenString for token in pred_print_out])
    # test data set
    example_test_generator = KerasBatchGenerator(test_data, step_input_size, 1, token_vocabulary_size,
                                                 skip_step=1)
    print("Test data:")
    num_predict = 10
    true_print_out = []
    pred_print_out = []
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, step_input_size - 1, :])
        true_print_out.append(index_to_token[test_data[step_input_size + i]])
        pred_print_out.append(index_to_token[predict_word])
    print("Actual")
    print([token.tokenString for token in true_print_out])
    print("Predicted")
    print([token.tokenString for token in pred_print_out])
