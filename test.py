import librosa


def test(filename):
    obj = {}
    audio, sr = librosa.load(filename)
    obj['instrument'] = sr
    return obj
