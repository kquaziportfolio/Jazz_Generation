import numpy as np

from music_utils import *
from preprocess import *
from tensorflow.keras.utils import to_categorical  # For some reason keras doesnt have this anymore

chords, agrams = get_musical_data("midi/original_metheny.mid")
corpus, tones, tids, idtones = get_corpus_data(agrams)
ntones = len(set(corpus))
n_a = 64
nvals_ = None


def load_music_utils():
    """Load music"""
    global nvals_

    chords, agrams = get_musical_data("midi/original_metheny.mid")
    corpus, tones, tids, idtones = get_corpus_data(agrams)
    ntones = len(set(corpus))
    X, Y, ntones = dataproc(corpus, tids, 60, 30)
    nvals_ = ntones

    return X, Y, ntones, idtones


load_music_utils()
x_init = np.zeros((1, 1, nvals_))
a_init = np.zeros((1, n_a))
c_init = np.zeros((1, n_a))


def generate_music(imodel, corpus=corpus, agrams=agrams, tones=tones, tids=tids, idtones=idtones, ty=10, mtries=1000,
                   div=0.5):
    """Use the model to generate music"""
    ostream = stream.Stream()

    coffest = 0.0
    nchords = int(len(chords) / 3)

    print("Predicting new values for a new set of chords")
    for i in range(1, nchords):
        cchords = stream.Voice()

        for j in chords[i]:
            cchords.insert((j.offset % 4), j)

        _, ids = pas(imodel)
        ids = list(ids.squeeze())
        pred = [idtones[j] for j in ids]

        ptones = "C,0.25 "  # middle c
        for k in range(len(pred) - 1):
            ptones += pred[k] + " "

        ptones += pred[-1]

        ptones = ptones.replace(" A", " C").replace(" X", " C")
        ptones = prune_grammar(ptones)
        sounds = clean_up_notes(prune_notes(unparse_grammar(ptones, cchords)))
        print("Generated %s sounds using the predicted values for the set of chords (\"%s\") and after pruning" % (
        len([k for k in sounds if isinstance(k, note.Note)]), i))
        for m in sounds:
            ostream.insert(coffest + m.offset, m)
        for mc in cchords:
            ostream.insert(coffest + mc.offset, mc)

        coffest += 4.0

    ostream.insert(0.0, tempo.MetronomeMark(number=130))  # 130 bpm

    ostream.write("midi", fp="music.midi")
    return ostream


def pas(imodel, xinit=x_init, ainit=a_init, cinit=c_init):
    """Predict the next value using the model"""
    pred = imodel.predict([xinit, ainit, cinit])
    ids = np.argmax(pred, axis=-1)
    results = to_categorical(ids, num_classes=nvals_)

    return results, ids
