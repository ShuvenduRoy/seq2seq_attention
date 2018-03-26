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


if __name__ == '__main__':
    # Load data
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

