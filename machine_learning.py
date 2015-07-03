#!/usr/bin/env python

"""Functions related to machine learning"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse
from itertools import izip
from collections import Counter
import random

import numpy as np
import sklearn
import matplotlib as plt
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

import feature_extraction
import instrument_parser


def confusion_str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrices"""
    ret_str = ''
    columnwidth = max([len(x) for x in labels]+[5])  # 5 is value length
    empty_cell = ' ' * columnwidth
    # Print header
    ret_str += '    ' + empty_cell
    for label in labels:
        ret_str += '%{0}s'.format(columnwidth) % label
    ret_str += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        ret_str += '    %{0}s'.format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            ret_str += cell
        ret_str += '\n'
    return ret_str


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues, save=False):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if save:
        plt.savefig(save)


def confusion(actual, predicted, plot=False, disp=False):
    confusion_labels = sorted(set(np.concatenate([actual, predicted])))
    confusion = confusion_matrix(actual, predicted, confusion_labels)
    if disp:
        print(confusion_str(confusion, confusion_labels))
    if plot:
        plot_confusion_matrix(confusion, confusion_labels)
    return confusion


def flatten(l):
    return [item for sublist in l for item in sublist]


def shape_labels(labels):
    Y = np.array(flatten(labels))
    return Y


def shape_features(tracks, feature_names):
    logging.debug('Shaping features/labels to sklearn format')
    features = []
    labels = []
    for track in tracks:
        feature = [track[f] for f in feature_names]
        feature = np.vstack(feature)
        features.append(feature)
        labels.append([track['label']] * feature.shape[1])

    X = np.concatenate([feature.T for feature in features])
    Y = np.array(flatten(labels))
    logging.debug('X: {}'.format(X.shape))
    logging.debug('Y: {}'.format(Y.shape))
    return X, Y


def train_features(features, labels):
    X = shape_features(features)
    Y = flatten(labels)
    clf = train(X, Y)
    return clf


def train(X, Y):
    #X = sklearn.preprocessing.normalize(X, norm='l1', axis=1, copy=True)
    logging.info('Training...')
    clf = sklearn.svm.LinearSVC(class_weight='auto')
    clf.fit(X, Y)
    return clf


def predict_features(features, clf):
    X = np.concatenate([feature.T for feature in features])
    return predict(X, clf)


def predict(X, clf):
    #X = sklearn.preprocessing.normalize(X, norm='l1', axis=1, copy=True)
    logging.info('Predicting...')
    predicted = clf.predict(X)
    return predicted


def prediction_per_track(tracks, features, predicted):
    i = 0
    for track, feature in izip(tracks, features):
        track_predictions = predicted[i:i+feature.shape[1]]
        most_common = Counter(track_predictions).most_common(1)[0][0]
        track.prediction = most_common
        i += feature.shape[1]
    return tracks


def cross_validate(tracks, features, labels, folds=5, shuffle=True):
    X = shape_features(features)
    Y = shape_labels(labels)
    kf = StratifiedKFold(Y, n_folds=folds, shuffle=shuffle)
    for train, test in kf:
        logging.info(train)
        logging.info(test)
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        clf = train(X_train, Y_train)
        predicted = predict(X_test, clf)
        cm = confusion(Y_test, predicted)
        print(cm)


def get_labels(tracks, features):
    labels = []
    for track, feature in izip(tracks, features):
        # Repeat instrument name for each mfcc sample
        labels.append([track.label] * feature.shape[1])
    return labels


def main(args):
    with instrument_parser.InstrumentParser() as instruments:
        if args.load_model:
            logging.info('Loading model into memory...')
            clf = joblib.load(args.load_model)
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
            mfccs = list(feature_extraction.get_mfccs(tracks))
            labels = get_labels(tracks, mfccs)
            clf = train(mfccs, labels)

        if args.save_model:
            logging.info('Saving model to disk...')
            joblib.dump(clf, args.save_model, compress=True)

        test_tracks = random.sample(tracks, 2)
        test_mfccs = list(feature_extraction.get_mfccs(test_tracks))
        predicted = predict(test_mfccs, clf)
        final_tracks = prediction_per_track(
            test_tracks,
            test_mfccs,
            predicted,
        )
        for track in final_tracks:
            logging.info(vars(track))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument stems from medleydb")
    parser.add_argument('action', type=str, choices={'train', 'predict'},
                        help='Action to take (Train or Predict)')
    parser.add_argument('-s', '--save_model', type=str, default=None,
                        help='Location to save pickled trained model')
    parser.add_argument('-l', '--load_model', type=str, default=None,
                        help='Location to load pickled model from')
    parser.add_argument('--save_features', type=str, default=None,
                        help='Location to save pickled features to')
    parser.add_argument('--load_features', type=str, default=None,
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
