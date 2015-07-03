#!/usr/bin/env python
"""Script to run Cross-Validation"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse

from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import instrument_parser
import feature_extraction
import machine_learning
import util


def report(actual, predicted, plot=False, save=False):
    labels = sorted(set(np.concatenate([actual, predicted])))
    confusion = confusion_matrix(actual, predicted, labels)
    confusion_string = machine_learning.confusion_str(confusion, labels)
    scores = classification_report(actual, predicted, target_names=labels)
    return '{}\n{}'.format(confusion_string, scores)


def track_report(tracks, plot=False, save=False):
    actual, predicted = [], []
    for track in tracks:
        actual.append(track['label'])
        predicted.append(track['prediction'])
    return report(actual, predicted, plot, save)


def cross_validate(tracks, feature_names, folds=5, shuffle=True):
    labels = [track['label'] for track in tracks]
    kf = StratifiedKFold(labels, n_folds=folds, shuffle=shuffle)
    for train, test in kf:
        train_tracks = [tracks[i] for i in train]
        test_tracks = [tracks[i] for i in test]
        X_train, Y_train = machine_learning.shape_features(train_tracks, feature_names)
        clf = machine_learning.train(X_train, Y_train)
        predicted_all = []
        Y_test_all = []
        for track in test_tracks:
            X_test, Y_test = machine_learning.shape_features([track], feature_names)
            predicted = machine_learning.predict(X_test, clf)
            track['sample_predictions'] = predicted
            track['prediction'] = util.most_common(predicted)
            predicted_all.extend(predicted)
            Y_test_all.extend(Y_test)
        scores = track_report(test_tracks, plot=True)
        print(scores)


def get_feature_names(args):
    feature_names = ['mfcc']
    if args.delta:
        feature_names.append('mfcc_delta')
    if args.delta_delta:
        feature_names.append('mfcc_delta_delta')
    return feature_names


def main(args):
    with instrument_parser.InstrumentParser() as instruments:
        if args.load_features:
            logging.info('Loading features into memory...')
            tracks = joblib.load(args.load_features)
        else:
            tracks = instruments.get_stems(
                args.min_sources,
                args.instruments,
                args.rm_silence,
                args.trim,
                args.instrument_count,
            )
            tracks = list(tracks)

            feature_extraction.set_track_mfccs(
                tracks,
                n_fft=args.n_fft,
                average=args.average,
                normalize=args.normalize,
                delta=args.delta,
                delta_delta=args.delta_delta,
            )
        feature_names = get_feature_names(args)
        cross_validate(tracks, feature_names, args.folds)

        if args.save_features:
            logging.info('Saving features to disk...')
            joblib.dump(tracks, args.save_features, compress=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument stems from medleydb")
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
    parser.add_argument('-k', '--folds', type=int, default=5,
                        help='Number of folds in kfold cross validation')
    parser.add_argument('-n', '--n_fft', type=int, default=2048,
                        help='FFT size of MFCCs')
    parser.add_argument('-a', '--average', type=int, default=None,
                        help='Number of seconds to average features over')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize MFCC feature vectors between 0 and 1')
    parser.add_argument('-d', '--delta', action='store_true',
                        help='Compute MFCC deltas')
    parser.add_argument('--delta_delta', action='store_true',
                        help='Compute MFCC delta-deltas')
    args = parser.parse_args()

    main(args)
