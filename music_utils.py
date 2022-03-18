import tensorflow.keras.backend as K
import numpy as np
import sys
import tensorflow as tf

from grammar import *
from tensorflow.keras.layers import RepeatVector
from music21 import *
from preprocess import *
from qa import *


def dataproc(corpus, valind, m=60, tx=30):
    """Splits corpus into sequences of tx values"""
    tx = tx  # python is strange
    nvals = len(set(corpus))
    X = np.zeros((m, tx, nvals), dtype=np.bool)
    Y = np.zeros((m, tx, nvals), dtype=np.bool)
    for i in range(m):
        ridx = np.random.choice(len(corpus)-tx)
        cdata = corpus[ridx:(ridx+tx)]
        for j in range(tx):
            idx = valind[cdata[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j, idx] = 1

    Y = Y.swapaxes(0, 1).tolist()
    return np.asarray(X), np.asarray(Y), nvals


def nextvalproc(model, nval, x, pas, ivals, agrams, dur, mtries=1000, temp=0.5):
    """Fix the first value"""

    # first note cant have crescendos or decrescendos, must not be rest
    if dur < 0.00001:
        tries = 0
        while nval.split(",")[0] == "R" or len(nval.split(",")) != 2:
            if tries >= mtries:
                # happens suprisingly often and messes up the model generation graph
                # print("Gave up on first note gen after", tries, "tries")
                nval = agrams[np.random.randint(0, len(agrams))].split(" ")[0]
            else:
                nval = pas(model, x, ivals, temp)

            tries += 1

    return nval


def seq2mat(seq, vids):
    """Convert a slice of the corpus to a numpy matrix of 1hot vectors"""

    sl = len(seq)
    x = np.zeros((1, sl, len(vids)))
    for i, val in enumerate(seq):
        if not(val in vids):
            print(val)

        x[0, i, vids[val]] = 1.
    return x


def onehot(x):
    """One hot"""
    x = K.argmax(x)
    x = tf.one_hot(x, 25)
    x = RepeatVector(1)(x)
    return x
