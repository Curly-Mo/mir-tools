#!/usr/bin/env python

"""Functions related to machine learning"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import os
import argparse
from itertools import izip
from collections import Counter
from time import time
import tempfile

import numpy as np
import sklearn
import matplotlib as plt
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

import feature_extraction
import util



def default_classifier():
    dot = os.path.dirname(os.path.realpath(__file__))
    DEFAULT = os.path.join(dot, 'data/default_svm.p')
    return load_classifier(DEFAULT)

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


def train_tracks(tracks, feature_names):
    X, Y = shape_features(tracks, feature_names)
    clf = train(X, Y)
    return clf


def train(X, Y):
    #X = sklearn.preprocessing.normalize(X, norm='l1', axis=1, copy=True)
    logging.info('Training...')
    clf = classifier()
    clf.fit(X, Y)
    return clf


def predict_file(path, remove_silence=True):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(path)[1]) as f:
        path = util.remove_silence(path, f.name, block=True)
        clf, feature_names = default_classifier()
        track = {
            'file_path': path,
            'label': 'unknown',
        }
        args = type('', (), {})
        args.n_fft = 4096
        args.average = 1
        args.normalize = True
        args.delta = True
        args.delta_delta = True
        args.save_features = False
        feature_extraction.set_features([track, ], args)

        prediction, predictions = predict_track(clf, track, feature_names)
    return prediction, predictions


def predict_track(clf, track, feature_names):
    predicted = predict_tracks(clf, [track, ], feature_names)
    track['sample_predictions'] = predicted
    prediction, predictions = util.most_common(predicted)
    track['prediction'] = prediction
    track['predictions'] = prediction
    return prediction, predictions


def predict_tracks(clf, tracks, feature_names):
    X, Y = shape_features(tracks, feature_names)
    return predict(X, clf)


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


def get_labels(tracks, features):
    labels = []
    for track, feature in izip(tracks, features):
        # Repeat instrument name for each mfcc sample
        labels.append([track.label] * feature.shape[1])
    return labels


def classifier():
    clf = sklearn.svm.LinearSVC(class_weight='auto')
    return clf


def save_classifier(path, clf, feature_names):
    classifier = {
        'clf': clf,
        'feature_names': feature_names,
    }
    logging.info('Saving model to disk...')
    joblib.dump(classifier, path, compress=True)


def load_classifier(path):
    logging.info('Loading model into memory...')
    classifier = joblib.load(path)
    return classifier['clf'], classifier['feature_names']


def train_main(args):
    if args.load_classifier:
        clf, feature_names = load_classifier(args.load_classifier)
    else:
        tracks = feature_extraction.load_tracks(args.label, args)
        feature_names = feature_extraction.get_feature_names(args)
        clf = train_tracks(tracks, feature_names)

    if args.save_classifier:
        save_classifier(args.save_classifier, clf, feature_names)


def predict_main(args):
    if args.load_classifier:
        clf, feature_names = load_classifier(args.load_classifier)
    else:
        tracks = feature_extraction.load_tracks(args.label, args)
        feature_names = feature_extraction.get_feature_names(args)
        clf = train_tracks(tracks, feature_names)

    if args.predict_files:
        predict_tracks = []
        for filepath in args.predict_files:
            prediction, predictions = predict_file(filepath, clf)
            print predictions
    else:
        predict_tracks = feature_extraction.load_tracks(args.label, args)
        feature_extraction.set_features(predict_tracks, args)

        for track in predict_tracks:
            prediction, predictions = predict_track(clf, track, feature_names)
            print predictions

    if args.save_classifier:
        save_classifier(args.save_classifier, clf, feature_names)


def main(args):
    start = time()
    if args.action == 'train':
        train_main(args)
    elif args.action == 'predict':
        predict_main(args)

    end = time()
    logging.info('Elapsed time: {}'.format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument stems from medleydb")
    parser.add_argument('action', type=str, choices={'train', 'predict'},
                        help='Action to take (Train or Predict)')
    parser.add_argument('label', type=str,
                        choices={'instrument', 'genre'},
                        help='Track label')
    parser.add_argument('-s', '--save_classifier', type=str, default=None,
                        help='Location to save pickled trained model')
    parser.add_argument('-l', '--load_classifier', type=str, default=None,
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
    parser.add_argument('--predict_files', nargs='*', default=None,
                        help='List of audio file paths to predict')
    args = parser.parse_args()

    main(args)
