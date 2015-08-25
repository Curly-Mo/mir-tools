#!/usr/bin/env python

"""Tools for extracting features"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import argparse

import librosa
from sklearn.externals import joblib
import numpy as np

import track_parser
import util


def get_mfccs(track, dc=False, n_fft=2048, average=None, normalize=False):
    audio, sr = librosa.load(track['file_path'])
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20, fmin=20, n_fft=n_fft)
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
                    delta=False, delta_delta=False, save=False):
    logging.info('Computing MFCCs...')
    for track in tracks:
        mfcc = get_mfccs(
            track,
            dc=dc,
            n_fft=n_fft,
            average=average,
            normalize=normalize
        )
        track['mfcc'] = mfcc
        if delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            track['mfcc_delta'] = mfcc_delta
        if delta_delta:
            mfcc_delta_delta = librosa.feature.delta(mfcc)
            track['mfcc_delta_delta'] = mfcc_delta_delta
    if save:
        logging.info('Saving tracks MFCCs to disk...')
        joblib.dump(tracks, save, compress=True)
    return tracks


def load_tracks(label, args):
    with track_parser.get_instance(label) as parser:
        tracks = parser.get_tracks(
            min_sources=args.min_sources,
            instruments=args.instruments,
            rm_silence=args.rm_silence,
            trim=args.trim,
            count=args.count,
        )
        tracks = list(tracks)

        set_track_mfccs(
            tracks,
            n_fft=args.n_fft,
            average=args.average,
            normalize=args.normalize,
            delta=args.delta,
            delta_delta=args.delta_delta,
            save=args.save_features,
        )
    return tracks


def main(args):
    if args.load_features:
        logging.info('Loading features into memory...')
        tracks = joblib.load(args.load_features)
    else:
        tracks = load_tracks(args)

    if args.save_features:
        logging.info('Saving features to disk...')
        joblib.dump(tracks, args.save_features, compress=True)

    logging.info(tracks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract features from medleydb")
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
    args = parser.parse_args()

    main(args)
