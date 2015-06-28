#!/usr/bin/env python
"""Script to run Cross-Validation"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse
import random

from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import instrument_parser
import feature_extraction
import machine_learning


def report(actual, predicted, plot=False, save=False):
    labels = sorted(set(np.concatenate([actual, predicted])))
    confusion = confusion_matrix(actual, predicted, labels)
    confusion_string = machine_learning.confusion_str(confusion, labels)
    scores = classification_report(actual, predicted, target_names=labels)
    return '{}\n{}'.format(confusion_string, scores)


def cross_validate(features, labels, folds=5, shuffle=True):
    X = machine_learning.shape_features(features)
    Y = machine_learning.shape_labels(labels)
    kf = StratifiedKFold(Y, n_folds=folds, shuffle=shuffle)
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        clf = machine_learning.train(X_train, Y_train)
        predicted = machine_learning.predict(X_test, clf)
        scores = report(Y_test, predicted, plot=True)
        print(scores)


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

            features = list(feature_extraction.get_mfccs(tracks))
            labels = machine_learning.get_labels(tracks, features)

        cross_validate(features, labels, args.folds)

        if args.save_features:
            logging.info('Saving features to disk...')
            joblib.dump((features, labels), args.save_features, compress=True)


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
    parser.add_argument('-k', '--folds', type=int, default=5,
                        help='Number of folds in kfold cross validation')
    args = parser.parse_args()

    main(args)
