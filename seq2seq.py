from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


def one_step_attention(a, s_prev):
    """
    Performs one step attention and give the context vector
    :param a: output of the bidirectional RNN layer
    :param s_prev: prev S
    :return: context C
    """
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    e = densor2(e)
    alphas = activator(e)
    context = dotor([alphas, a])

    return context


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s, ), name='s0')
    c0 = Input(shape=(n_s, ), name='c0')
    s = s0
    c = c0

    # initialize empty list of outputs
    outputs = []

    # Step1: Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step2: Iterate for Ty time
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)

        outputs.append(out)

    model = Model([X, s0, c0], outputs)
    return model


if __name__ == '__main__':
    """ Load and preprocess data """
    m = 10000
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

    # print some sample data
    print(dataset[:10])

    # Length of input sequence (with text this data len can be long)
    Tx = 30
    # Length of output sequence
    Ty = 10

    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    print("X.shape:", X.shape)          # X.shape: (10000, 30)
    print("Y.shape:", Y.shape)          # Y.shape: (10000, 10)
    print("Xoh.shape:", Xoh.shape)      # Xoh.shape: (10000, 30, 37)  //One hot encoded
    print("Yoh.shape:", Yoh.shape)      # Yoh.shape: (10000, 10, 11)  //One hot encoded

    """ Define the model """
    # Define shared layer as global variable
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation='tanh')
    densor2 = Dense(1, activation='relu')
    activator = Activation(softmax, name='attention_weights')
    dotor = Dot(axes=1)

    n_a = 32
    n_s = 64
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(len(machine_vocab), activation=softmax)

    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.summary()

    # define the compiler
    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=0.01)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)

    """ Test the model """
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
        prediction = model.predict([source.reshape(1, 30, 37), s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]
        # print("raw outputs: ", output)

        print("source:", example)
        print("output:", ''.join(output))