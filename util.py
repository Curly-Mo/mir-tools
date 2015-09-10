from itertools import izip
from collections import Counter
import subprocess
import logging


def grouper(iterable, n, padvalue=None):
    """
    Group interable into chunks of size n
    Removes last group if shorter than n
    grouper('abcdefg', 3) --> (('a','b','c'), ('d','e','f'))
    """
    return izip(*[iter(iterable)]*n)


def most_common(values):
    total = len(values)
    most_common_list = Counter(values).most_common()
    most_common_list = [
        (pair[0], round(100*pair[1]/float(total)))
        for pair in most_common_list
    ]
    most_common = most_common_list[0][0]
    return most_common, most_common_list


def remove_silence(input_path, output_path, block=True):
    logging.info('Removing silence from: ' + str(input_path) + ' to: ' + str(output_path) + '\n')
    sox_args = ['sox', input_path, '-c', '1', output_path, 'silence', '1', '0.1', '0.1%', '-1', '0.1', '0.1%']
    process_handle = subprocess.Popen(sox_args, stderr=subprocess.PIPE)
    if block:
        process_handle.communicate()
    return output_path


# This is no longer needed
# def load_audio(object_or_path):
#     try:
#         audio, sr = librosa.load(object_or_path)
#     except:
#         audio, sr = parse_audio(object_or_path)
#     return audio, sr
#
#
# def parse_audio(file_object, sr=22050, mono=True, offset=0.0, duration=None,
#                 dtype=np.float32):
#     """Load an audio file object as a floating point time series.
#     Imitates librosa's load function, but for already opened file objects
#
#     Parameters
#     ----------
#     file_object : opened file object
#         Any format supported by `audioread` will work.
#     sr   : number > 0 [scalar]
#         target sampling rate
#         'None' uses the native sampling rate
#     mono : bool
#         convert signal to mono
#     offset : float
#         start reading after this time (in seconds)
#     duration : float
#         only load up to this much audio (in seconds)
#     dtype : numeric type
#         data type of `y`
#     Returns
#     -------
#     y    : np.ndarray [shape=(n,) or (2, n)]
#         audio time series
#     sr   : number > 0 [scalar]
#         sampling rate of `y`
#     """
# 
#     import audioread_object
#     y = []
#     with audioread_object.audio_open(file_object) as input_file:
#         sr_native = input_file.samplerate
# 
#         s_start = int(np.round(sr_native * offset)) * input_file.channels
# 
#         if duration is None:
#             s_end = np.inf
#         else:
#             s_end = s_start + (int(np.round(sr_native * duration))
#                                * input_file.channels)
# 
#         n = 0
# 
#         for frame in input_file:
#             frame = librosa.util.buf_to_float(frame, dtype=dtype)
#             n_prev = n
#             n = n + len(frame)
# 
#             if n < s_start:
#                 # offset is after the current frame
#                 # keep reading
#                 continue
# 
#             if s_end < n_prev:
#                 # we're off the end.  stop reading
#                 break
# 
#             if s_end < n:
#                 # the end is in this frame.  crop.
#                 frame = frame[:s_end - n_prev]
# 
#             if n_prev <= s_start <= n:
#                 # beginning is in this frame
#                 frame = frame[(s_start - n_prev):]
# 
#             # tack on the current frame
#             y.append(frame)
# 
#     if y:
#         y = np.concatenate(y)
# 
#         if input_file.channels > 1:
#             y = y.reshape((-1, 2)).T
#             if mono:
#                 y = librosa.to_mono(y)
# 
#         if sr is not None:
#             if y.ndim > 1:
#                 y = np.vstack([librosa.resample(yi, sr_native, sr) for yi in y])
#             else:
#                 y = librosa.resample(y, sr_native, sr)
# 
#         else:
#             sr = sr_native
# 
#     # Final cleanup for dtype and contiguity
#     y = np.ascontiguousarray(y, dtype=dtype)
# 
#     return (y, sr)
