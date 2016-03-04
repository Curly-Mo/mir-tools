#!/usr/bin/env python

"""Tools for extracting features"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse
from time import time

import librosa
from sklearn.externals import joblib
import numpy as np

import track_parser
import util


def get_mfccs(track, dc=False, n_fft=2048, average=None, normalize=False,
              n_mfcc=20, fmin=20, fmax=None, hop_length=512, n_mels=128, **kwargs):
    audio, sr = librosa.load(track['file_path'])
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
                                fmin=fmin, fmax=fmax, hop_length=hop_length,
                                n_mels=n_mels)
    if not dc:
        mfcc = mfcc[1:]
    if normalize:
        # Normalize each feature vector between 0 and 1
        mfcc = mfcc - mfcc.min(axis=0)
        mfcc = mfcc / mfcc.max(axis=0)
    if average and average > 0:
        samples = sr * average / n_fft
        chunks = util.grouper(mfcc.T, samples)
        averaged_chunk = [np.mean(group, axis=0) for group in chunks]
        mfcc = np.array(averaged_chunk).T
    return mfcc


def set_track_mfccs(tracks, dc=False, n_fft=2048, average=None, normalize=False,
                    feature_names=None, save_features=False, **kwargs):
    logging.info('Computing MFCCs...')
    for track in tracks:
        if 'mfcc' in feature_names:
            mfcc = get_mfccs(
                track,
                dc=dc,
                n_fft=n_fft,
                average=average,
                normalize=normalize,
                **kwargs
            )
            track['mfcc'] = mfcc
        if 'mfcc_delta' in feature_names:
            mfcc_delta = librosa.feature.delta(mfcc)
            track['mfcc_delta'] = mfcc_delta
        if 'mfcc_delta_delta' in feature_names:
            mfcc_delta_delta = librosa.feature.delta(mfcc)
            track['mfcc_delta_delta'] = mfcc_delta_delta
    if save_features:
        logging.info('Saving tracks MFCCs to disk...')
        args = locals()
        args.pop('tracks', None)
        args.pop('track', None)
        args.pop('mfcc', None)
        args.pop('mfcc_delta', None)
        args.pop('mfcc_delta_delta', None)
        print args
        joblib.dump([tracks, args], save_features, compress=True)
    return tracks


def set_features(tracks, **kwargs):
    mfcc_features = ('mfcc', 'mfcc_delta', 'mfcc_delta_delta')
    if any(x in kwargs['feature_names'] for x in mfcc_features):
        set_track_mfccs(tracks, **kwargs)


def load_tracks(label, **kwargs):
    if kwargs.get('load_features'):
        logging.info('Loading features into memory...')
        [tracks, kwargs] = joblib.load(kwargs['load_features'])
    else:
        with track_parser.get_instance(label) as parser:
            tracks = parser.get_tracks(**kwargs)
            tracks = list(tracks)

            set_features(tracks, **kwargs)
    return [tracks, kwargs]


def main(**kwargs):
    start = time()
    if kwargs['load_features']:
        logging.info('Loading features into memory...')
        tracks, args = joblib.load(kwargs['load_features'])
    else:
        tracks, _ = load_tracks(**kwargs)

    if kwargs['save_features']:
        logging.info('Saving features to disk...')
        joblib.dump([tracks, kwargs], kwargs['save_features'], compress=True)

    logging.info(tracks)
    end = time()
    logging.info('Elapsed time: {}'.format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract features from medleydb")
    parser.add_argument('label', type=str, choices={'instrument', 'genre'},
                        help='Track label')
    parser.add_argument('-s', '--save_features', type=str, default=None,
                        help='Location to save pickled features to')
    parser.add_argument('-l', '--load_features', type=str, default=None,
                        help='Location to load pickled features from')
    parser.add_argument('-m', '--min_sources', nargs='*', default=10,
                        help='Min sources required for instrument selection')
    parser.add_argument('-i', '--instruments', nargs='*', default=None,
                        help='List of instruments to extract')
    parser.add_argument('-c', '--instrument_count', type=int, default=None,
                        help='Max number of tracks for each instrument')
    parser.add_argument('-r', '--rm_silence', action='store_true',
                        help='Remove silence from audio files')
    parser.add_argument('-t', '--trim', type=int, default=None,
                        help='Trim audio files to this length (in seconds)')
    parser.add_argument('-f', '--feature_names', nargs='+', default=None,
                        choices=['mfcc', 'mfcc_delta', 'mfcc_delta_delta'],
                        required=True,
                        help='List of features names to use')
    args = parser.parse_args()

    main(**vars(args))
