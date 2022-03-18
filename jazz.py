# Requires Tensorflow 2

from data_utils import *
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

epochs = 1000
epochs = int(sys.argv[1]) if len(sys.argv) == 2 else epochs

# GPU stuff
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the music
X, Y, nvals, ivals = load_music_utils()
print("Shape of X:", X.shape)
print("Number of training examples:", X.shape[0])
print("Tx (length of sequence):", X.shape[1])
print("Number of unique values:", nvals)
print("Shape of Y:", Y.shape)

n_a = 64

# Layers for the model
reshapor = Reshape((1, nvals))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(nvals, activation="softmax")


def genmodel(tx, na, n_vals):
    """Generates the actual model"""
    X = Input(shape=(tx, n_vals))
    a0 = Input(shape=(na,), name="a0")
    c0 = Input(shape=(na,), name="c0")
    a = a0
    c = c0

    outputs = []
    for t in range(tx):
        x = Lambda(lambda x: x[:, t, :])(X)
        x = reshapor(x)

        a, _, c = LSTM_cell(x, initial_state=[a, c])

        outputs.append(densor(a))

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


# Creates an instance of the model and trains it
model = genmodel(tx=30, na=n_a, n_vals=nvals)
opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

seqmod = model.fit([X, a0, c0], list(Y), epochs=epochs, verbose=2)


def imodelgen(LSTM_cell, densor, nvals=nvals, n_a=n_a, ty=50):
    """Creates an inference model"""
    x0 = Input(shape=(1, nvals))
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0
    x = x0

    outputs = []
    for t in range(ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        outputs.append(densor(a))
        x = Lambda(onehot)(densor(a))

    imodel = Model(inputs=[x0, a0, c0], outputs=outputs)
    return imodel


# Create and instance of the instance model
imodel = imodelgen(LSTM_cell, densor, nvals, n_a, 100)
x_init = np.zeros((1, 1, nvals))
a_init = np.zeros((1, n_a))
c_init = np.zeros((1, n_a))


def pas(imod, xinit=x_init, ainit=a_init, cinit=c_init):
    """Predict and sample"""
    print(xinit.shape)
    print(ainit.shape)
    print(cinit.shape)
    pred = imod.predict([xinit, ainit, cinit])
    ids = np.argmax(np.array(pred), axis=-1)
    results = to_categorical(ids, num_classes=nvals)

    return ids, results


# Get results
# results, ids = pas(imodel, x_init, a_init, c_init)
# print("np.argmax(results[12]) =", np.argmax(results[12]))
# print("np.argmax(results[17]) =", np.argmax(results[17]))
# print("list(indices[12:18]) =", list(ids[12:18]))

# Generate the music and print the loss
if __name__ == "__main__":
    ostream = generate_music(imodel)

# Debug GPU
print(device_lib.list_local_devices())
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Model Final Loss:", seqmod.history["loss"][-1])
