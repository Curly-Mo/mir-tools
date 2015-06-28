#!/usr/bin/env python

"""Main Script"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse
import random

import librosa
from sklearn.externals import joblib

import instrument_parser
import machine_learning


def get_mfccs(tracks, dc=False):
    logging.info('Computing MFCCs...')
    for track in tracks:
        audio, sr = librosa.load(track.file_path)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20, fmin=20)
        if not dc:
            mfcc = mfcc[1:]
        yield mfcc


def main(args):
    with instrument_parser.InstrumentParser() as instruments:
        if args.load_features:
            logging.info('Loading features into memory...')
            features, labels = joblib.load(args.load_features)
        else:
            tracks = instruments.get_stems(
                args.min_sources,
                args.instruments,
                args.rm_silence,
                args.trim
            )
            tracks = list(tracks)
            if args.test_subset:
                tracks = random.sample(tracks, args.test_subset)

            features = list(get_mfccs(tracks))
            labels = machine_learning.get_labels(tracks, features)

        if args.save_features:
            logging.info('Saving features to disk...')
            joblib.dump((features, labels), args.save_features, compress=True)

        print(features)
        print(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument stems from medleydb")
    parser.add_argument('-s', '--save_features', type=str, default=None,
                        help='Location to save pickled features to')
    parser.add_argument('-l', '--load_features', type=str, default=None,
                        help='Location to load pickled features from')
    parser.add_argument('-u', '--test_subset', type=int, default=None,
                        help='Run a test on small subset of data')
    parser.add_argument('-m', '--min_sources', nargs='*', default=10,
                        help='Min sources required for instrument selection')
    parser.add_argument('-i', '--instruments', nargs='*', default=None,
                        help='List of instruments to extract')
    parser.add_argument('-r', '--rm_silence', action='store_true',
                        help='Remove silence from audio files')
    parser.add_argument('-t', '--trim', type=int, default=None,
                        help='Trim all audio files down to this length (in seconds)')
    args = parser.parse_args()

    main(args)
