#!/usr/bin/env python
"""Script to run Cross-Validation"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse
from time import time

import scipy
import sklearn
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import feature_extraction
import machine_learning
import util


def report(actual, predicted, plot=False, save=False):
    labels = np.unique(np.concatenate([actual, predicted]))
    confusion = confusion_matrix(actual, predicted, labels)
    confusion_string = machine_learning.confusion_str(confusion, labels)
    scores = classification_report(actual, predicted, target_names=labels)
    return '{}\n{}'.format(confusion_string, scores)


def sample_report(tracks, plot=False, save=False):
    actual, predicted = [], []
    for track in tracks:
        predicted.extend(track['sample_predictions'])
        actual.extend([track['label']] * len(track['sample_predictions']))
    return report(actual, predicted, plot, save)


def track_report(tracks, plot=False, save=False):
    actual, predicted = [], []
    for track in tracks:
        actual.append(track['label'])
        predicted.append(track['prediction'])
    return report(actual, predicted, plot, save)


def best_svm(tracks, feature_names, n_iter=200, save=False):
    clf = sklearn.svm.LinearSVC(class_weight='auto')
    X, Y = machine_learning.shape_features(tracks, feature_names)
    param_dist = {
        'C': scipy.stats.expon(scale=1000),
        'class_weight': ['auto', None],
        'loss': ['squared_hinge'],
        'penalty': ['l1', 'l2'],
        'dual': [False],
        'tol': scipy.stats.expon(scale=0.1),
    }
    logging.info('Optimizing parameters: {}'.format(param_dist))
    random_search = sklearn.grid_search.RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter
    )
    random_search.fit(X, Y)
    for score in random_search.grid_scores_:
        print(score)
    print('Best Score: {}'.format(random_search.best_score_))
    print('Best Params: {}'.format(random_search.best_params_))
    if save:
        logging.info('Saving classifier to disk...')
        joblib.dump(random_search.best_estimator_, save, compress=True)
    return random_search.best_estimator_


def cross_val_score(tracks, feature_names, folds=5):
    X, Y = machine_learning.shape_features(tracks, feature_names)
    clf = sklearn.svm.LinearSVC(class_weight='auto')
    scores = cross_validation.cross_val_score(
        clf,
        X,
        Y,
        cv=folds,
        scoring='f1_weighted'
    )
    return scores


def kfold(tracks, feature_names, folds=5, shuffle=True):
    labels = [track['label'] for track in tracks]
    kf = cross_validation.StratifiedKFold(labels, n_folds=folds, shuffle=shuffle)
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
        yield test_tracks


def get_feature_names(args):
    feature_names = ['mfcc']
    if args.delta:
        feature_names.append('mfcc_delta')
    if args.delta_delta:
        feature_names.append('mfcc_delta_delta')
    return feature_names


def main(args):
    start = time()
    if args.load_features:
        logging.info('Loading features into memory...')
        tracks = joblib.load(args.load_features)
    else:
        tracks = feature_extraction.load_tracks(args.label, args)

    feature_names = get_feature_names(args)

    if args.action == 'kfold':
        folds = kfold(tracks, feature_names, args.folds)
        for tracks in folds:
            scores = sample_report(tracks, plot=True)
            print(scores)
            scores = track_report(tracks, plot=True)
            print(scores)
    elif args.action == 'cross_val_score':
        scores = cross_val_score(tracks, feature_names, folds=args.folds)
        print(scores)
    elif args.action == 'optimize':
        clf = best_svm(tracks, feature_names, args.save_classifier)
        print(clf)

    end = time()
    logging.info('Elapsed time: {}'.format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument stems from medleydb")
    parser.add_argument('action', type=str,
                        choices={'kfold', 'cross_val_score', 'optimize'},
                        help='Action to take')
    parser.add_argument('label', type=str,
                        choices={'instrument', 'genre'},
                        help='Track label')
    parser.add_argument('-s', '--save_features', type=str, default=None,
                        help='Location to save pickled features to')
    parser.add_argument('-l', '--load_features', type=str, default=None,
                        help='Location to load pickled features from')
    parser.add_argument('-m', '--min_sources', nargs='*', default=10,
                        help='Min sources required for instrument selection')
    parser.add_argument('-i', '--instruments', nargs='*', default=None,
                        help='List of instruments to extract')
    parser.add_argument('-g', '--genress', nargs='*', default=None,
                        help='List of genress to extract')
    parser.add_argument('-c', '--count', type=int, default=None,
                        help='Max number of tracks for each label')
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
    parser.add_argument('--save_classifier', type=str, default=None,
                        help='Location to save pickled classifier to')
    args = parser.parse_args()

    main(args)
