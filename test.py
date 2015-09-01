import sys
if sys.version_info >= (3, 0):
    from past import autotranslate
    autotranslate(['librosa'])
import librosa


def test(filename):
    obj = {}
    audio, sr = librosa.load(filename)
    obj['instrument'] = sr
    return obj
